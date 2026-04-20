"""
scripts/14_vis_base_pose_full_preprocess.py

在 base_pose 规范朝向下跑一遍与 01_preprocess_modelnet40 相同的后续管线（HPR、
dropout、噪声、密度重采样、随机 SE3、bbox+PCA），仅可视化、不写盘。

与正式预处理的唯一区别：用 base_poses.json 中的 R_align 替换 Rotation.random() 的 R_aug。

示例：
  python scripts/14_vis_base_pose_full_preprocess.py --stem airplane_0627 --view 0 --seed 0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import open3d as o3d

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))


def _load_preprocess_module():
    path = os.path.join(_SCRIPT_DIR, "01_preprocess_modelnet40.py")
    spec = importlib.util.spec_from_file_location("preprocess_modelnet40", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _to_pcd(pts: np.ndarray, color: List[float]) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    pcd.paint_uniform_color(color)
    return pcd


def _layout_x(geoms: List[o3d.geometry.PointCloud], margin: float = 0.35) -> None:
    """沿 +X 排开，避免不同坐标系叠在一起。"""
    x0 = 0.0
    for g in geoms:
        pts = np.asarray(g.points, dtype=np.float64)
        if pts.size == 0:
            continue
        mn, mx = pts.min(axis=0), pts.max(axis=0)
        dx = x0 - mn[0]
        g.translate((dx, 0.0, 0.0))
        pts2 = np.asarray(g.points, dtype=np.float64)
        x0 = float(pts2[:, 0].max()) + margin


def run_one_sample(
    pre: Any,
    mesh_path: str,
    R_align: np.ndarray,
    *,
    num_cad_sample: int,
    num_obs: int,
    num_gt: int,
    min_keep: int,
    missing_rate_min: float,
    missing_rate_max: float,
    hpr_sphere_r: float,
    hpr_radius: float,
    hpr_radius_factor: float,
    fib_jitter: float,
    density_alpha: float,
    noise_std: float,
    se3_rot_max_deg: float,
    se3_trans_min: float,
    se3_trans_max: float,
    pca_axis: str,
    view_idx: int,
    num_views: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    mesh = pre._read_off_mesh(mesh_path)
    if not mesh.has_triangles():
        raise ValueError(f"无三角面片: {mesh_path}")

    pcd = mesh.sample_points_uniformly(number_of_points=num_cad_sample)
    gt_full, gt_nrm = pre._normalize_point_cloud(pcd)

    R_aug = np.asarray(R_align, dtype=np.float32).reshape(3, 3)
    gt_rot = (gt_full @ R_aug.T).astype(np.float32)
    n_rot = (gt_nrm @ R_aug.T).astype(np.float32)

    gt_min, gt_max = gt_rot.min(axis=0), gt_rot.max(axis=0)
    extent = float(np.max(gt_max - gt_min))

    _cam_to_far = float(hpr_sphere_r) + 1.0
    hpr_r_effective = max(float(hpr_radius), float(hpr_radius_factor) * _cam_to_far)

    cameras = pre.fibonacci_sphere_cameras(num_views, hpr_sphere_r, jitter=fib_jitter)
    if not (0 <= view_idx < num_views):
        raise ValueError(f"view_idx 需在 [0, {num_views})")
    camera_pos = cameras[view_idx]

    pcd_hpr = o3d.geometry.PointCloud()
    pcd_hpr.points = o3d.utility.Vector3dVector(gt_rot.astype(np.float64))
    _, pt_map = pcd_hpr.hidden_point_removal(camera_pos.astype(np.float64), hpr_r_effective)
    pt_map = np.asarray(pt_map, dtype=np.int64)
    vis_pts = gt_rot[pt_map]
    vis_nrm = n_rot[pt_map]

    mr = float(np.random.uniform(missing_rate_min, missing_rate_max))
    mr = min(mr, max(0.0, 1.0 - min_keep / max(vis_pts.shape[0], 1)))
    if np.random.rand() > 0.5:
        cor, cor_n = pre.apply_depth_dropout(vis_pts, vis_nrm, camera_pos=camera_pos, missing_rate=mr)
    else:
        theta_l, phi_l = np.random.uniform(0, 2 * np.pi), np.random.uniform(0, np.pi)
        ld = np.array(
            [
                np.sin(phi_l) * np.cos(theta_l),
                np.sin(phi_l) * np.sin(theta_l),
                np.cos(phi_l),
            ],
            dtype=np.float32,
        )
        cor, cor_n = pre.apply_specular_dropout(vis_pts, vis_nrm, camera_pos=camera_pos, light_dir=ld, missing_rate=mr)

    cor, cor_n = pre._ensure_min_points(cor, cor_n, min_keep)
    cor = cor + np.random.normal(0, noise_std, cor.shape).astype(np.float32)

    p_corrupted = pre.density_weighted_resample(cor, camera_pos, num_obs, alpha=density_alpha)
    gt_resampled = pre.resample(gt_rot, num_gt)

    R_se3 = pre.random_rotation_matrix(se3_rot_max_deg)
    t_se3 = pre.random_translation(se3_trans_min, se3_trans_max, extent)

    p_obs = (p_corrupted @ R_se3.T + t_se3[None, :]).astype(np.float32)
    gt_w = (gt_resampled @ R_se3.T + t_se3[None, :]).astype(np.float32)

    p_norm, C_bbox, scale = pre.normalize_by_bbox(p_obs)
    p_in, R_pca, mu_pca = pre.pca_align(p_norm, target_axis=pca_axis)
    gt_norm = ((gt_w - C_bbox[None, :]) / scale).astype(np.float32)
    gt_out = pre.apply_pca_rigid(gt_norm, R_pca, mu_pca)

    dbg = {
        "camera_pos": camera_pos,
        "hpr_r_effective": hpr_r_effective,
        "missing_rate": mr,
        "n_visible_hpr": int(vis_pts.shape[0]),
        "n_after_dropout": int(cor.shape[0]),
        "extent": extent,
    }
    return p_obs, p_in, gt_out, gt_rot, dbg


def main():
    ap = argparse.ArgumentParser(description="base_pose + 完整预处理管线（仅可视化）")
    ap.add_argument(
        "--base-json",
        default=os.path.join(_PROJECT_ROOT, "data", "processed", "modelnet40", "base_poses.json"),
    )
    ap.add_argument("--stem", required=True, help="mesh stem，如 airplane_0627")
    ap.add_argument("--raw-dir", default=os.path.join(_PROJECT_ROOT, "data", "raw", "ModelNet40"))
    ap.add_argument("--view", type=int, default=0, help="Fibonacci 视角索引 [0, num_views)")
    ap.add_argument("--num-views", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--num-cad-sample", type=int, default=16384)
    ap.add_argument("--num-obs", type=int, default=2048)
    ap.add_argument(
        "--num-gt",
        type=int,
        default=16384,
        help="与 01_preprocess_modelnet40 --num-gt 一致（默认 16384）",
    )
    ap.add_argument("--min-keep", type=int, default=1024)
    ap.add_argument("--missing-rate-min", type=float, default=0.1)
    ap.add_argument("--missing-rate-max", type=float, default=0.4)
    ap.add_argument("--hpr-sphere-r", type=float, default=3.0)
    ap.add_argument("--hpr-radius", type=float, default=100.0)
    ap.add_argument("--hpr-radius-factor", type=float, default=25.0)
    ap.add_argument("--fib-jitter", type=float, default=0.15)
    ap.add_argument("--density-alpha", type=float, default=1.5)
    ap.add_argument("--noise-std", type=float, default=0.002)
    ap.add_argument("--se3-rot-max-deg", type=float, default=15.0)
    ap.add_argument("--se3-trans-min", type=float, default=0.5)
    ap.add_argument("--se3-trans-max", type=float, default=2.0)
    ap.add_argument("--pca-axis", default="z", choices=("x", "y", "z"))
    args = ap.parse_args()

    if not os.path.isfile(args.base_json):
        print(f"找不到 {args.base_json}", file=sys.stderr)
        sys.exit(1)

    with open(args.base_json, encoding="utf-8") as f:
        data = json.load(f)

    stem = args.stem
    if stem not in data.get("poses", {}) or stem not in data.get("meta", {}):
        print(f"base_poses.json 中无键: {stem}", file=sys.stderr)
        sys.exit(1)

    T = np.asarray(data["poses"][stem], dtype=np.float64).reshape(4, 4)
    R_align = T[:3, :3].astype(np.float32)

    meta = data["meta"][stem]
    mesh_path = meta.get("source_path")
    if not mesh_path or not os.path.isfile(mesh_path):
        cat, sp = meta.get("category", ""), meta.get("split", "")
        cand = os.path.join(args.raw_dir, str(cat), str(sp), f"{stem}.off")
        if os.path.isfile(cand):
            mesh_path = cand
    if not mesh_path or not os.path.isfile(mesh_path):
        print(f"找不到 mesh: {stem}", file=sys.stderr)
        sys.exit(1)

    np.random.seed(args.seed)
    pre = _load_preprocess_module()

    p_obs, p_in, gt_out, gt_rot_canon, dbg = run_one_sample(
        pre,
        mesh_path,
        R_align,
        num_cad_sample=args.num_cad_sample,
        num_obs=args.num_obs,
        num_gt=args.num_gt,
        min_keep=args.min_keep,
        missing_rate_min=args.missing_rate_min,
        missing_rate_max=args.missing_rate_max,
        hpr_sphere_r=args.hpr_sphere_r,
        hpr_radius=args.hpr_radius,
        hpr_radius_factor=args.hpr_radius_factor,
        fib_jitter=args.fib_jitter,
        density_alpha=args.density_alpha,
        noise_std=args.noise_std,
        se3_rot_max_deg=args.se3_rot_max_deg,
        se3_trans_min=args.se3_trans_min,
        se3_trans_max=args.se3_trans_max,
        pca_axis=args.pca_axis,
        view_idx=args.view,
        num_views=args.num_views,
    )

    print(
        f"{stem}  view={args.view}  HPR可见={dbg['n_visible_hpr']}  "
        f"dropout后={dbg['n_after_dropout']}  mr={dbg['missing_rate']:.3f}  "
        f"hpr_r={dbg['hpr_r_effective']:.2f}"
    )

    # 窗口 1：规范朝向下完整 GT（base_pose 后、HPR 前）
    o3d.visualization.draw_geometries(
        [_to_pcd(gt_rot_canon, [0.35, 0.55, 0.85])],
        window_name=f"{stem} | canonical GT (after R_align), N={gt_rot_canon.shape[0]}",
        width=1200,
        height=700,
    )

    # 窗口 2：obs（世界系，含 SE3） / input+gt（同一 PCA 空间，左右分栏）
    g_obs = _to_pcd(p_obs, [0.9, 0.25, 0.15])
    g_in = _to_pcd(p_in, [0.2, 0.45, 0.95])
    g_gt = _to_pcd(gt_out, [0.25, 0.85, 0.35])
    _layout_x([g_obs, g_in, g_gt], margin=0.4)
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.15, origin=[0, 0, 0])
    o3d.visualization.draw_geometries(
        [g_obs, g_in, g_gt, frame],
        window_name=f"{stem} | L:obs  M:input  R:gt (model space) | view {args.view}",
        width=1400,
        height=720,
    )


if __name__ == "__main__":
    main()
