"""
ITODD 3D-to-2D Pose Refinement test (pure math / differentiable projection).

Dependencies:
    torch, open3d, cv2, numpy, scipy, json

Usage (default paths auto-adapt to this repo):
    python scripts/08_itodd_pose_refine_test.py

    python scripts/08_itodd_pose_refine_test.py \
        --itodd-root ITODD \
        --model-path models/models/obj_000001.ply \
        --camera-path base/itoddmv/camera_cam0.json \
        --scene-gt-path val/val/000001/scene_gt_3dlong.json \
        --image-dir val/val/000001/gray_cam0 \
        --obj-id 1 --num-iters 100
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as SciRot


def nan_to_num_compat(x: torch.Tensor, nan: float = 0.0, posinf: float | None = None, neginf: float | None = None) -> torch.Tensor:
    """
    torch.nan_to_num 的兼容实现：旧版 torch 可能没有 nan_to_num。
    """
    if hasattr(torch, "nan_to_num"):
        return torch.nan_to_num(x, nan=nan, posinf=posinf, neginf=neginf)
    y = x
    # replace NaN
    y = torch.where(y == y, y, torch.tensor(nan, dtype=y.dtype, device=y.device))
    # replace +/-inf
    if posinf is not None:
        y = torch.where(y == float("inf"), torch.tensor(posinf, dtype=y.dtype, device=y.device), y)
    if neginf is not None:
        y = torch.where(y == float("-inf"), torch.tensor(neginf, dtype=y.dtype, device=y.device), y)
    return y


def resolve_existing_path(root: str, candidates: List[str]) -> str:
    for rel in candidates:
        p = os.path.join(root, rel)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"No candidate path exists under {root}: {candidates}")


def load_camera_K(camera_json_path: str, device: torch.device) -> torch.Tensor:
    with open(camera_json_path, "r", encoding="utf-8") as f:
        cam = json.load(f)
    fx, fy, cx, cy = cam["fx"], cam["fy"], cam["cx"], cam["cy"]
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float32)
    return torch.from_numpy(K).to(device)


def sample_model_points(model_ply_path: str, num_points: int, device: torch.device) -> torch.Tensor:
    mesh = o3d.io.read_triangle_mesh(model_ply_path)
    if mesh.is_empty():
        raise RuntimeError(f"Failed to load mesh: {model_ply_path}")
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    pts = np.asarray(pcd.points, dtype=np.float32)  # model coordinates
    return torch.from_numpy(pts).to(device)


def load_scene_gt(scene_gt_path: str) -> Dict[str, List[Dict]]:
    with open(scene_gt_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_scene_gt_info(scene_gt_info_path: str) -> Dict[str, List[Dict]]:
    with open(scene_gt_info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_first_instance(scene_gt: Dict[str, List[Dict]], obj_id: int) -> Tuple[int, int, Dict]:
    im_ids = sorted(int(k) for k in scene_gt.keys())
    for im_id in im_ids:
        anns = scene_gt[str(im_id)]
        for inst_id, ann in enumerate(anns):
            if int(ann["obj_id"]) == int(obj_id):
                return im_id, inst_id, ann
    raise ValueError(f"obj_id={obj_id} not found in scene_gt.")


def build_T_from_ann(ann: Dict) -> np.ndarray:
    R = np.array(ann["cam_R_m2c"], dtype=np.float64).reshape(3, 3)
    t = np.array(ann["cam_t_m2c"], dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def make_perturbed_pose(T_gt: np.ndarray, trans_mm: float = 10.0, rot_deg: float = 5.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)

    # Fixed-norm translation perturbation.
    d_t = rng.normal(size=3)
    d_t = d_t / (np.linalg.norm(d_t) + 1e-12) * trans_mm

    # Fixed-angle rotation perturbation (left-multiply in camera frame).
    axis = rng.normal(size=3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    dR = SciRot.from_rotvec(axis * np.deg2rad(rot_deg)).as_matrix()

    T_icp = np.eye(4, dtype=np.float64)
    T_icp[:3, :3] = dR @ T_gt[:3, :3]
    T_icp[:3, 3] = T_gt[:3, 3] + d_t
    return T_icp


def pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> Tuple[float, float]:
    # rotation error in degree
    dR = T_est[:3, :3] @ T_gt[:3, :3].T
    trace = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    rot_err_deg = float(np.rad2deg(np.arccos(trace)))

    # translation error in mm
    trans_err_mm = float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
    return trans_err_mm, rot_err_deg


def rodrigues_torch(rotvec: torch.Tensor) -> torch.Tensor:
    # rotvec: (3,)
    theta = torch.norm(rotvec)
    I = torch.eye(3, device=rotvec.device, dtype=rotvec.dtype)
    if float(theta.detach().cpu()) < 1e-8:
        # first-order approx around zero
        wx, wy, wz = rotvec[0], rotvec[1], rotvec[2]
        K = torch.tensor(
            [[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]],
            device=rotvec.device,
            dtype=rotvec.dtype,
        )
        return I + K

    k = rotvec / theta
    kx, ky, kz = k[0], k[1], k[2]
    K = torch.stack(
        [
            torch.stack([torch.tensor(0.0, device=rotvec.device), -kz, ky]),
            torch.stack([kz, torch.tensor(0.0, device=rotvec.device), -kx]),
            torch.stack([-ky, kx, torch.tensor(0.0, device=rotvec.device)]),
        ]
    )
    R = I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)
    return R


def project_points_torch(points_m: torch.Tensor, R: torch.Tensor, t: torch.Tensor, K: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    points_m: (N,3) in model frame
    R: (3,3), t: (3,), K: (3,3)
    return:
      uv: (N,2) pixel coords
      z: (N,)
    """
    pts_cam = (R @ points_m.T).T + t[None, :]  # (N,3), mm
    z = pts_cam[:, 2]
    proj = (K @ pts_cam.T).T
    uv = proj[:, :2] / proj[:, 2:3].clamp(min=1e-6)
    return uv, z


def build_distance_field_from_gray(gray_path: str, canny1: int = 80, canny2: int = 160) -> Tuple[np.ndarray, np.ndarray]:
    # ITODD 的 tif 通常是 16-bit；直接 IMREAD_GRAYSCALE 可能导致动态范围极窄，Canny 近乎空边缘。
    img_u = cv2.imread(gray_path, cv2.IMREAD_UNCHANGED)
    if img_u is None:
        raise FileNotFoundError(f"Cannot read image: {gray_path}")

    if img_u.ndim == 3:
        img_u = cv2.cvtColor(img_u, cv2.COLOR_BGR2GRAY)

    # Robust normalize to 8-bit for edge detection
    if img_u.dtype == np.uint16 or img_u.dtype == np.int16:
        img_f = img_u.astype(np.float32)
        lo, hi = np.percentile(img_f, [1.0, 99.0])
        if hi <= lo + 1e-6:
            lo, hi = float(img_f.min()), float(img_f.max()) + 1e-6
        img_n = np.clip((img_f - lo) / (hi - lo), 0.0, 1.0)
        img8 = (img_n * 255.0).astype(np.uint8)
    else:
        # ensure uint8
        if img_u.dtype != np.uint8:
            img8 = cv2.normalize(img_u, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            img8 = img_u

    img = img8
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {gray_path}")

    edges = cv2.Canny(img, canny1, canny2)
    # Fallback: if canny returns empty, use gradient magnitude threshold
    if int((edges > 0).sum()) == 0:
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        thr = float(np.percentile(mag, 95.0))
        edges = (mag > thr).astype(np.uint8) * 255

    # distance to nearest edge: distanceTransform computes distance to nearest zero pixel.
    # so edge pixels should be zero in input.
    inv = np.where(edges > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)
    # Clip + normalize to keep loss scale stable and avoid inf
    max_dist = 50.0
    dist = np.minimum(dist, max_dist) / max_dist
    return edges, dist


def build_distance_field_from_mask(mask_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用可见 mask 的边界作为“真实轮廓”，构建距离场（更适合 Render-and-Compare 精对齐）。
    返回:
      edges (uint8 0/255), dist (float32 in [0,1])
    """
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {mask_path}")
    m_bin = (m > 0).astype(np.uint8) * 255
    # Morphological gradient gives a clean boundary even on binary masks
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edges = cv2.morphologyEx(m_bin, cv2.MORPH_GRADIENT, k)

    inv = np.where(edges > 0, 0, 255).astype(np.uint8)
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 3).astype(np.float32)
    max_dist = 50.0
    dist = np.minimum(dist, max_dist) / max_dist
    return edges, dist


def sample_dist_with_grid_sample(dist_field: torch.Tensor, uv: torch.Tensor, H: int, W: int) -> torch.Tensor:
    """
    dist_field: (1,1,H,W)
    uv: (N,2) in pixel coordinates
    return: sampled distances (N,)
    """
    x = 2.0 * (uv[:, 0] / (W - 1.0)) - 1.0
    y = 2.0 * (uv[:, 1] / (H - 1.0)) - 1.0
    grid = torch.stack([x, y], dim=-1).view(1, -1, 1, 2)
    sampled = F.grid_sample(
        dist_field,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )  # (1,1,N,1)
    return sampled.view(-1)


def optimize_pose_2d(
    points_m: torch.Tensor,
    K: torch.Tensor,
    dist_field: torch.Tensor,
    T_init: np.ndarray,
    H: int,
    W: int,
    num_iters: int = 100,
    lr: float = 3e-2,
    reg_w_rot: float = 1e-2,
    reg_w_trans: float = 1e-4,
    uv_offset: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[np.ndarray, List[float]]:
    device = points_m.device
    dtype = torch.float32

    R0 = SciRot.from_matrix(T_init[:3, :3]).as_rotvec().astype(np.float32)
    t0 = T_init[:3, 3].astype(np.float32)

    r_opt = torch.tensor(R0, dtype=dtype, device=device, requires_grad=True)
    t_opt = torch.tensor(t0, dtype=dtype, device=device, requires_grad=True)
    r_ref = torch.tensor(R0, dtype=dtype, device=device)
    t_ref = torch.tensor(t0, dtype=dtype, device=device)

    optimizer = torch.optim.Adam([r_opt, t_opt], lr=lr)
    loss_hist: List[float] = []
    off_u = float(uv_offset[0])
    off_v = float(uv_offset[1])
    printed_diag = False

    for i in range(num_iters):
        optimizer.zero_grad()

        R = rodrigues_torch(r_opt)
        uv, z = project_points_torch(points_m, R, t_opt, K)
        if off_u != 0.0 or off_v != 0.0:
            uv = uv - torch.tensor([off_u, off_v], device=device, dtype=uv.dtype)[None, :]

        valid = (z > 1.0) & (uv[:, 0] >= 0.0) & (uv[:, 0] < (W - 1.0)) & (uv[:, 1] >= 0.0) & (uv[:, 1] < (H - 1.0))
        if not printed_diag:
            with torch.no_grad():
                print(
                    f"[Diag] uv_offset=({off_u:.1f},{off_v:.1f})  "
                    f"uv_x=[{uv[:,0].min().item():.1f},{uv[:,0].max().item():.1f}]  "
                    f"uv_y=[{uv[:,1].min().item():.1f},{uv[:,1].max().item():.1f}]  "
                    f"crop_WxH=({W}x{H})  valid={int(valid.sum().item())}/{uv.shape[0]}"
                )
            printed_diag = True
        if valid.any():
            d_vals = sample_dist_with_grid_sample(dist_field, uv[valid], H, W)
            d_vals = nan_to_num_compat(d_vals, nan=1.0, posinf=1.0, neginf=1.0)
            d_vals = d_vals.clamp(min=0.0, max=1.0)
            # robust loss on distances
            data_loss = torch.mean(torch.sqrt(d_vals + 1e-6))
        else:
            data_loss = torch.tensor(1e3, device=device, dtype=dtype, requires_grad=True)

        # small-update prior around T_init to prevent drifting to background edges
        reg_loss = reg_w_rot * torch.sum((r_opt - r_ref) ** 2) + reg_w_trans * torch.sum((t_opt - t_ref) ** 2)
        loss = data_loss + reg_loss

        loss.backward()
        optimizer.step()
        loss_hist.append(float(loss.detach().cpu()))

        if (i + 1) % 10 == 0 or i == 0:
            print(f"[Iter {i+1:03d}/{num_iters}] loss={loss_hist[-1]:.6f} valid={int(valid.sum().item())}")

    with torch.no_grad():
        Rf = rodrigues_torch(r_opt).detach().cpu().numpy().astype(np.float64)
        tf = t_opt.detach().cpu().numpy().astype(np.float64)
    T_final = np.eye(4, dtype=np.float64)
    T_final[:3, :3] = Rf
    T_final[:3, 3] = tf
    return T_final, loss_hist


def main():
    parser = argparse.ArgumentParser(description="ITODD differentiable 3D-to-2D pose refinement test")
    parser.add_argument("--itodd-root", type=str, default="/home/csy/SnowflakeNet_FPFH_ICP/ITODD")
    parser.add_argument("--model-path", type=str, default="models/obj_000001.ply")
    parser.add_argument("--camera-path", type=str, default="base/camera_cam0.json")
    parser.add_argument("--scene-gt-path", type=str, default="val/000000/scene_gt.json")
    parser.add_argument("--image-dir", type=str, default="val/000000/gray")
    parser.add_argument("--obj-id", type=int, default=1)
    parser.add_argument("--num-points", type=int, default=2048)
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--trans-perturb-mm", type=float, default=2.0)
    parser.add_argument("--rot-perturb-deg", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--canny1", type=int, default=80)
    parser.add_argument("--canny2", type=int, default=160)
    parser.add_argument("--out-dir", type=str, default="outputs/itodd_pose_refine")
    parser.add_argument("--mask-dir", type=str, default="val/val/000001/mask_visib_3dlong")
    parser.add_argument("--scene-gt-info-path", type=str, default="val/val/000001/scene_gt_info_3dlong.json")
    parser.add_argument("--use-mask", action="store_true", help="使用 mask_visib 的边界构建距离场（推荐）")
    parser.add_argument("--crop-pad", type=int, default=20, help="使用 bbox_visib 裁剪距离场的边界扩展像素")
    parser.add_argument("--reg-w-rot", type=float, default=1e-2, help="旋转正则权重（围绕 T_init 的小更新先验）")
    parser.add_argument("--reg-w-trans", type=float, default=1e-4, help="平移正则权重（mm 单位）")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    root = os.path.abspath(args.itodd_root)

    # Auto-resolve for this repo's ITODD layout, while keeping user-provided paths configurable.
    model_path = os.path.join(root, args.model_path)
    if not os.path.exists(model_path):
        model_path = resolve_existing_path(root, ["models/models/obj_000001.ply", "models/obj_000001.ply"])

    scene_gt_path = os.path.join(root, args.scene_gt_path)
    if not os.path.exists(scene_gt_path):
        scene_gt_path = resolve_existing_path(root, ["val/val/000001/scene_gt_3dlong.json", "val/000000/scene_gt.json"])

    # Decide sensor *after* resolving real scene_gt_path.
    prefer_3dlong = "3dlong" in os.path.basename(scene_gt_path)

    # If user didn't override camera/image args (still defaults) and GT is 3dlong,
    # auto-switch to the matching intrinsics/images to keep projection consistent.
    if prefer_3dlong and args.camera_path == "base/camera_cam0.json":
        args.camera_path = "base/itoddmv/camera_3dlong.json"
    if prefer_3dlong and args.image_dir == "val/000000/gray":
        args.image_dir = "val/val/000001/gray_3dlong"

    camera_path = os.path.join(root, args.camera_path)
    if not os.path.exists(camera_path):
        cam_candidates = ["base/itoddmv/camera_cam0.json", "base/camera_cam0.json"]
        if prefer_3dlong:
            cam_candidates = ["base/itoddmv/camera_3dlong.json"] + cam_candidates
        camera_path = resolve_existing_path(root, cam_candidates)

    scene_gt_info_path = os.path.join(root, args.scene_gt_info_path)
    if not os.path.exists(scene_gt_info_path):
        # optional: allow missing
        scene_gt_info_path = resolve_existing_path(root, ["val/val/000001/scene_gt_info_3dlong.json"]) if os.path.exists(os.path.join(root, "val/val/000001/scene_gt_info_3dlong.json")) else scene_gt_info_path

    image_dir = os.path.join(root, args.image_dir)
    if not os.path.isdir(image_dir):
        img_candidates = ["val/val/000001/gray_cam0", "val/000000/gray"]
        if prefer_3dlong:
            img_candidates = ["val/val/000001/gray_3dlong"] + img_candidates
        image_dir = resolve_existing_path(root, img_candidates)

    mask_dir = os.path.join(root, args.mask_dir)
    if not os.path.isdir(mask_dir):
        # optional: allow missing
        mask_dir = resolve_existing_path(root, ["val/val/000001/mask_visib_3dlong"])

    print("=== Config ===")
    print(f"model     : {model_path}")
    print(f"camera    : {camera_path}")
    print(f"scene_gt  : {scene_gt_path}")
    print(f"image_dir : {image_dir}")
    print(f"obj_id    : {args.obj_id}")
    print(f"device    : {args.device}")
    print("==============")

    device = torch.device(args.device)
    points_m = sample_model_points(model_path, args.num_points, device)
    K = load_camera_K(camera_path, device)
    scene_gt = load_scene_gt(scene_gt_path)
    scene_gt_info = load_scene_gt_info(scene_gt_info_path) if os.path.exists(scene_gt_info_path) else {}

    # Step 2: dynamic target lookup from JSON
    im_id, inst_id, ann = find_first_instance(scene_gt, args.obj_id)
    T_gt = build_T_from_ann(ann)

    # Resolve image file by im_id (prefer .tif then .png)
    image_name = f"{im_id:06d}"
    gray_path = os.path.join(image_dir, f"{image_name}.tif")
    if not os.path.exists(gray_path):
        gray_path = os.path.join(image_dir, f"{image_name}.png")
    if not os.path.exists(gray_path):
        raise FileNotFoundError(f"No gray image for im_id={im_id}: {image_dir}/{image_name}.(tif|png)")

    # Step 3: construct degraded initialization
    T_icp = make_perturbed_pose(
        T_gt,
        trans_mm=args.trans_perturb_mm,
        rot_deg=args.rot_perturb_deg,
        seed=args.seed,
    )

    # Step 4: edge + distance field
    uv_off = (0.0, 0.0)
    if args.use_mask:
        mask_path = os.path.join(mask_dir, f"{im_id:06d}_{inst_id:06d}.png")
        edges, dist_np = build_distance_field_from_mask(mask_path)
        # Optional crop by bbox_visib from scene_gt_info (speeds up & reduces background influence)
        if str(im_id) in scene_gt_info and inst_id < len(scene_gt_info[str(im_id)]):
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
    else:
        edges, dist_np = build_distance_field_from_gray(gray_path, canny1=args.canny1, canny2=args.canny2)

    H, W = dist_np.shape[:2]
    dist_t = torch.from_numpy(dist_np).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Step 5: optimization
    T_final, loss_hist = optimize_pose_2d(
        points_m=points_m,
        K=K,
        dist_field=dist_t,
        T_init=T_icp,
        H=H,
        W=W,
        num_iters=args.num_iters,
        lr=args.lr,
        reg_w_rot=args.reg_w_rot,
        reg_w_trans=args.reg_w_trans,
        uv_offset=uv_off,
    )

    # Step 6: evaluation
    init_trans_mm, init_rot_deg = pose_error(T_icp, T_gt)
    final_trans_mm, final_rot_deg = pose_error(T_final, T_gt)

    # Save loss curve + debug artifacts
    out_dir = os.path.join("/home/csy/SnowflakeNet_FPFH_ICP", args.out_dir) if not os.path.isabs(args.out_dir) else args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    stem = f"obj{args.obj_id:06d}_im{im_id:06d}"
    np.save(os.path.join(out_dir, f"{stem}_loss.npy"), np.array(loss_hist, dtype=np.float32))
    cv2.imwrite(os.path.join(out_dir, f"{stem}_edges.png"), edges)
    if args.use_mask:
        cv2.imwrite(os.path.join(out_dir, f"{stem}_mask_edges.png"), edges)

    # Plot loss curve if matplotlib is available
    try:
        import matplotlib.pyplot as plt  # type: ignore
        plt.figure(figsize=(6, 4), dpi=150)
        plt.plot(loss_hist, linewidth=1.5)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.title(f"ITODD pose refine loss ({stem})")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{stem}_loss.png"))
        plt.close()
        loss_plot_path = os.path.join(out_dir, f"{stem}_loss.png")
    except Exception:
        loss_plot_path = os.path.join(out_dir, f"{stem}_loss.npy")

    print("\n=== Results ===")
    print(f"Target image id       : {im_id:06d}")
    print(f"Target gray image     : {gray_path}")
    print(f"Initial perturb error : trans={init_trans_mm:.3f} mm, rot={init_rot_deg:.3f} deg")
    print(f"Final optimized error : trans={final_trans_mm:.3f} mm, rot={final_rot_deg:.3f} deg")
    print("Loss history (first 10):", [round(v, 6) for v in loss_hist[:10]])
    print("Loss history (last 10) :", [round(v, 6) for v in loss_hist[-10:]])
    print(f"Saved: {loss_plot_path}")
    print(f"Saved: {os.path.join(out_dir, f'{stem}_edges.png')}")


if __name__ == "__main__":
    main()

