"""
scripts/09_itodd_pose_refine_validate.py

两种验证（不在终端刷屏，全部写 log 文件）：
1) 2D overlay：将优化前/后投影点叠加到 mask_edges 上，保存 PNG。
2) 多随机扰动统计：同一 (obj_id, im_id, inst_id) 下跑 N 次不同 seed，输出误差统计 CSV + log。

依赖：
  torch, open3d, cv2, numpy, scipy, json, (可选 matplotlib)

用法示例：
  python scripts/09_itodd_pose_refine_validate.py --use-mask --trials 20
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import importlib.util
import os
import sys
from typing import Any, Dict, Tuple

import cv2
import numpy as np
import torch


def load_module_from_path(py_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, py_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import {name} from {py_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def write_overlay_png(
    out_path: str,
    edges: np.ndarray,
    uv_before: np.ndarray,
    uv_after: np.ndarray,
    color_before=(0, 0, 255),
    color_after=(255, 128, 0),
):
    """
    edges: uint8, HxW (0/255)
    uv_*: Nx2 in pixel coords of the same crop (already applied uv_offset)
    """
    if edges.ndim != 2:
        raise ValueError("edges must be HxW grayscale")
    H, W = edges.shape
    canvas = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    def draw_points(uv: np.ndarray, color):
        if uv.size == 0:
            return
        u = np.round(uv[:, 0]).astype(np.int32)
        v = np.round(uv[:, 1]).astype(np.int32)
        m = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u, v = u[m], v[m]
        for x, y in zip(u.tolist(), v.tolist()):
            canvas[y, x] = color

    draw_points(uv_before, color_before)
    draw_points(uv_after, color_after)
    cv2.imwrite(out_path, canvas)


def project_np(points_m: np.ndarray, K: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    points_m: (N,3) model
    K: (3,3)
    T: (4,4) cam_T_obj
    returns uv (M,2), mask_valid (N,)
    """
    R = T[:3, :3].astype(np.float64)
    t = T[:3, 3].astype(np.float64)
    X = (R @ points_m.astype(np.float64).T + t.reshape(3, 1)).T
    valid = X[:, 2] > 1.0
    Xv = X[valid]
    uvw = (K.astype(np.float64) @ Xv.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return uv.astype(np.float32), valid


def main():
    repo_root = "/home/csy/SnowflakeNet_FPFH_ICP"
    rc_path = os.path.join(repo_root, "scripts", "08_itodd_pose_refine_test.py")
    rc = load_module_from_path(rc_path, "itodd_refine")

    parser = argparse.ArgumentParser(description="ITODD pose refine validation (overlay + multi-trials)")
    parser.add_argument("--itodd-root", default=os.path.join(repo_root, "ITODD"))
    parser.add_argument("--obj-id", type=int, default=1)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--trans-perturb-mm", type=float, default=10.0)
    parser.add_argument("--rot-perturb-deg", type=float, default=5.0)
    parser.add_argument("--use-mask", action="store_true")
    parser.add_argument("--crop-pad", type=int, default=20)
    parser.add_argument("--reg-w-rot", type=float, default=1e-2)
    parser.add_argument("--reg-w-trans", type=float, default=1e-4)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed0", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out-dir", default=os.path.join(repo_root, "outputs", "itodd_pose_refine_eval"))
    args = parser.parse_args()

    out_dir = ensure_dir(args.out_dir if os.path.isabs(args.out_dir) else os.path.join(repo_root, args.out_dir))
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(out_dir, f"validate_obj{args.obj_id:06d}_{ts}.log")
    csv_path = os.path.join(out_dir, f"trials_obj{args.obj_id:06d}_{ts}.csv")

    with open(log_path, "w", encoding="utf-8") as log_f, contextlib.redirect_stdout(log_f), contextlib.redirect_stderr(log_f):
        print("=== ITODD pose refine validation ===")
        print("log:", log_path)
        print("csv:", csv_path)
        print("out_dir:", out_dir)

        device = torch.device(args.device)
        root = os.path.abspath(args.itodd_root)

        # Reuse 08's auto-resolve logic by calling its helpers directly
        model_path = rc.resolve_existing_path(root, ["models/models/obj_000001.ply", "models/obj_000001.ply"]).replace("obj_000001.ply", f"obj_{args.obj_id:06d}.ply")
        if not os.path.exists(model_path):
            # fallback to obj_000001 if specific not found
            model_path = rc.resolve_existing_path(root, [f"models/models/obj_{args.obj_id:06d}.ply", "models/models/obj_000001.ply"])

        scene_gt_path = rc.resolve_existing_path(root, ["val/val/000001/scene_gt_3dlong.json"])
        scene_gt_info_path = rc.resolve_existing_path(root, ["val/val/000001/scene_gt_info_3dlong.json"])
        mask_dir = rc.resolve_existing_path(root, ["val/val/000001/mask_visib_3dlong"])
        camera_path = rc.resolve_existing_path(root, ["base/itoddmv/camera_3dlong.json"])

        print("model:", model_path)
        print("camera:", camera_path)
        print("scene_gt:", scene_gt_path)
        print("scene_gt_info:", scene_gt_info_path)
        print("mask_dir:", mask_dir)
        print("device:", device)

        points_m_t = rc.sample_model_points(model_path, args.num_points, device)
        points_m = points_m_t.detach().cpu().numpy().astype(np.float32)
        K_t = rc.load_camera_K(camera_path, device)
        K = K_t.detach().cpu().numpy().astype(np.float32)

        scene_gt = rc.load_scene_gt(scene_gt_path)
        scene_gt_info = rc.load_scene_gt_info(scene_gt_info_path)
        im_id, inst_id, ann = rc.find_first_instance(scene_gt, args.obj_id)
        T_gt = rc.build_T_from_ann(ann)

        print(f"target im_id={im_id:06d} inst_id={inst_id:06d}")

        # Build DT field from mask (and crop)
        uv_off = (0.0, 0.0)
        mask_path = os.path.join(mask_dir, f"{im_id:06d}_{inst_id:06d}.png")
        edges, dist_np = rc.build_distance_field_from_mask(mask_path)
        bbox = scene_gt_info[str(im_id)][inst_id].get("bbox_visib", None)
        if bbox is not None:
            x, y, w, h = map(int, bbox)
            pad = int(args.crop_pad)
            x0 = max(0, x - pad)
            y0 = max(0, y - pad)
            x1 = min(dist_np.shape[1], x + w + pad)
            y1 = min(dist_np.shape[0], y + h + pad)
            dist_np = dist_np[y0:y1, x0:x1]
            edges = edges[y0:y1, x0:x1]
            uv_off = (float(x0), float(y0))
        H, W = dist_np.shape[:2]
        dist_t = torch.from_numpy(dist_np).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        # 1) Overlay validation on a single run (seed=seed0)
        seed = args.seed0
        T_init = rc.make_perturbed_pose(T_gt, trans_mm=args.trans_perturb_mm, rot_deg=args.rot_perturb_deg, seed=seed)
        T_final, loss_hist = rc.optimize_pose_2d(
            points_m=points_m_t,
            K=K_t,
            dist_field=dist_t,
            T_init=T_init,
            H=H,
            W=W,
            num_iters=args.num_iters,
            lr=args.lr,
            reg_w_rot=args.reg_w_rot,
            reg_w_trans=args.reg_w_trans,
            uv_offset=uv_off,
        )
        init_t_mm, init_r_deg = rc.pose_error(T_init, T_gt)
        fin_t_mm, fin_r_deg = rc.pose_error(T_final, T_gt)
        print("--- Single-run ---")
        print(f"seed={seed}  init: {init_t_mm:.3f}mm {init_r_deg:.3f}deg  final: {fin_t_mm:.3f}mm {fin_r_deg:.3f}deg")
        print(f"loss: {loss_hist[0]:.6f} -> {loss_hist[-1]:.6f}")

        uv_b, _ = project_np(points_m, K, T_init)
        uv_a, _ = project_np(points_m, K, T_final)
        # crop offset to local coords
        uv_b = uv_b - np.array(uv_off, dtype=np.float32)[None, :]
        uv_a = uv_a - np.array(uv_off, dtype=np.float32)[None, :]
        overlay_path = os.path.join(out_dir, f"overlay_obj{args.obj_id:06d}_im{im_id:06d}_{ts}.png")
        write_overlay_png(overlay_path, edges, uv_b, uv_a)
        print("overlay saved:", overlay_path)

        # 2) Multi-trial stats
        print("--- Multi-trials ---")
        rows = []
        for k in range(args.trials):
            sd = args.seed0 + k
            Ti = rc.make_perturbed_pose(T_gt, trans_mm=args.trans_perturb_mm, rot_deg=args.rot_perturb_deg, seed=sd)
            Tf, lh = rc.optimize_pose_2d(
                points_m=points_m_t,
                K=K_t,
                dist_field=dist_t,
                T_init=Ti,
                H=H,
                W=W,
                num_iters=args.num_iters,
                lr=args.lr,
                reg_w_rot=args.reg_w_rot,
                reg_w_trans=args.reg_w_trans,
                uv_offset=uv_off,
            )
            it, ir = rc.pose_error(Ti, T_gt)
            ft, fr = rc.pose_error(Tf, T_gt)
            rows.append((sd, it, ir, ft, fr, lh[0], lh[-1]))
            print(f"trial {k+1:02d}/{args.trials} seed={sd}  init({it:.2f}mm,{ir:.2f}deg) -> final({ft:.2f}mm,{fr:.2f}deg)")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["seed", "init_trans_mm", "init_rot_deg", "final_trans_mm", "final_rot_deg", "loss0", "lossN"])
            w.writerows(rows)

        arr = np.array(rows, dtype=np.float64)
        init_t = arr[:, 1]
        init_r = arr[:, 2]
        final_t = arr[:, 3]
        final_r = arr[:, 4]
        print("CSV saved:", csv_path)
        print("Summary:")
        print(f"  init  trans mean={init_t.mean():.3f} std={init_t.std():.3f}  rot mean={init_r.mean():.3f} std={init_r.std():.3f}")
        print(f"  final trans mean={final_t.mean():.3f} std={final_t.std():.3f}  rot mean={final_r.mean():.3f} std={final_r.std():.3f}")
        print(f"  improved trans: {(final_t < init_t).sum()}/{args.trials}")
        print(f"  improved rot  : {(final_r < init_r).sum()}/{args.trials}")

    # Do not print to terminal; only tell the user where logs are.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

