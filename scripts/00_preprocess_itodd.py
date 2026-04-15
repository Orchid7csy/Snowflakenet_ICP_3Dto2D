"""
scripts/00_preprocess_itodd.py

纯 CAD 合成 ITODD completion 训练集生成器。

数据源：28 个 ITODD CAD 模型（ITODD/models/models/obj_*.ply）。
增强管线（每个 aug_id 内部）：
  1) 动态微观采样：每次 aug 重新从 mesh 采样 16384 点（不复用上一次的点位）
  2) 全局 SO(3) 随机旋转
  3) Fibonacci 球面视角覆盖：生成 num_views 个相机坐标（带 jitter）
  4) HPR 2.5D 可见性 → depth/specular dropout → 下限保护
  5) 非均匀深度密度重采样（近密远疏）→ 2048
  6) bbox 中心化 + PCA 对齐；GT 同步同一变换 → 2048

train/test 按 (obj_id, aug_id) 级别划分（同一 aug 的所有视角在同一 split）。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import tempfile
from typing import List, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.transforms import apply_depth_dropout, apply_specular_dropout


# ═══════════════════════════ shared utilities ═══════════════════════════


def fibonacci_sphere_cameras(n: int, r: float, jitter: float = 0.0) -> np.ndarray:
    """
    在半径 r 的球面上用 Fibonacci 序列均匀放置 n 个相机位置。
    jitter: 对 phi/theta 的随机扰动幅度（弧度），0 则为纯确定性。
    返回 (n, 3) float32。
    """
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    indices = np.arange(n, dtype=np.float64)
    phi = np.arccos(1.0 - 2.0 * (indices + 0.5) / n)
    theta = 2.0 * np.pi * indices / golden
    if jitter > 0:
        phi += np.random.uniform(-jitter, jitter, size=n)
        theta += np.random.uniform(-jitter, jitter, size=n)
    phi = np.clip(phi, 1e-6, np.pi - 1e-6)
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def density_weighted_resample(
    points: np.ndarray, camera_pos: np.ndarray, n: int, alpha: float = 1.5,
) -> np.ndarray:
    """
    非均匀深度密度重采样：距相机越近的点被选中概率越高，模拟工业 RGBD 近密远疏特性。
    alpha 越大，近端概率越集中。
    """
    if points.shape[0] == 0:
        return np.zeros((n, 3), dtype=np.float32)
    d = np.linalg.norm(points - camera_pos[None, :], axis=1)
    d_max = d.max() + 1e-8
    w = ((d_max - d) / d_max) ** alpha + 1e-8
    w /= w.sum()
    idx = np.random.choice(points.shape[0], size=n, replace=(points.shape[0] < n), p=w)
    return points[idx].astype(np.float32)


def normalize_by_bbox(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    c = np.asarray(pcd.get_axis_aligned_bounding_box().get_center(), dtype=np.float32)
    return (points - c[None, :]).astype(np.float32), c


def _orthonormal_frame(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    v = v / n
    tmp = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(v, tmp)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0])
    e2 = np.cross(v, tmp)
    e2 /= np.linalg.norm(e2) + 1e-12
    e3 = np.cross(v, e2)
    return np.stack([v, e2, e3], axis=1)


def pca_align(points: np.ndarray, target_axis: str = "z", min_ratio: float = 1e-4):
    pcd = o3d.geometry.PointCloud()
    pts = points.astype(np.float64)
    pcd.points = o3d.utility.Vector3dVector(pts)
    mu, cov = pcd.compute_mean_and_covariance()
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    cov = np.asarray(cov, dtype=np.float64).reshape(3, 3)

    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    lam1, lam2 = float(evals[order[0]]), float(evals[order[1]]) if len(order) > 1 else 0.0
    if lam1 < 1e-12 or (lam1 - lam2) / (lam1 + 1e-12) < min_ratio:
        return pts.astype(np.float32), np.eye(3, dtype=np.float32), mu.astype(np.float32)

    v_main = evecs[:, order[0]].astype(np.float64)
    v_main /= np.linalg.norm(v_main) + 1e-12
    target_map = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
    target = np.array(target_map[target_axis.lower()], dtype=np.float64)
    if np.dot(v_main, target) < 0:
        v_main = -v_main

    B = _orthonormal_frame(v_main)
    T = _orthonormal_frame(target)
    R = T @ B.T
    if np.linalg.det(R) < 0:
        T[:, 1] *= -1.0
        R = T @ B.T
    aligned = ((R @ (pts - mu).T).T + mu).astype(np.float32)
    return aligned, R.astype(np.float32), mu.astype(np.float32)


def apply_pca_rigid(points: np.ndarray, R: np.ndarray, mu: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float64)
    R64, mu64 = R.astype(np.float64), mu.astype(np.float64).reshape(1, 3)
    return ((R64 @ (pts - mu64).T).T + mu64).astype(np.float32)


def resample(points: np.ndarray, n: int) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((n, 3), dtype=np.float32)
    if points.shape[0] >= n:
        return points[np.random.choice(points.shape[0], n, replace=False)].astype(np.float32)
    return points[np.random.choice(points.shape[0], n, replace=True)].astype(np.float32)


def estimate_normals(points: np.ndarray, radius: float = 0.05, max_nn: int = 30) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=int(max_nn)))
    return np.asarray(pcd.normals, dtype=np.float32)


def _ensure_min_points(points: np.ndarray, normals: np.ndarray, min_keep: int):
    n = points.shape[0]
    if n >= min_keep or n == 0:
        return points, normals
    idx = np.random.choice(n, min_keep - n, replace=True)
    return np.concatenate([points, points[idx]]).astype(np.float32), np.concatenate([normals, normals[idx]]).astype(np.float32)


def _assign_split(key: str, train_ratio: float, seed: int) -> str:
    h = hashlib.blake2b(f"{seed}:{key}".encode(), digest_size=8).digest()
    u = int.from_bytes(h, "little") / float(2**64)
    return "train" if u < train_ratio else "test"


# ═══════════════════════════ main ═══════════════════════════


def main():
    ap = argparse.ArgumentParser(description="ITODD CAD 合成 completion 预处理")
    ap.add_argument("--itodd-root", default="/home/csy/SnowflakeNet_FPFH_ICP/ITODD")
    ap.add_argument("--models-dir", default="models/models")
    ap.add_argument("--out-root", default="/home/csy/SnowflakeNet_FPFH_ICP/data/processed/itodd")
    ap.add_argument("--num-aug-per-model", type=int, default=125)
    ap.add_argument("--num-views", type=int, default=8)
    ap.add_argument("--num-cad-sample", type=int, default=16384)
    ap.add_argument("--num-obs", type=int, default=2048)
    ap.add_argument("--num-gt", type=int, default=2048)
    ap.add_argument("--min-keep", type=int, default=1024)
    ap.add_argument("--missing-rate-min", type=float, default=0.3)
    ap.add_argument("--missing-rate-max", type=float, default=0.7)
    ap.add_argument("--hpr-sphere-r", type=float, default=3.0)
    ap.add_argument("--hpr-radius", type=float, default=100.0)
    ap.add_argument("--fib-jitter", type=float, default=0.15)
    ap.add_argument("--density-alpha", type=float, default=1.5)
    ap.add_argument("--normal-radius", type=float, default=0.05)
    ap.add_argument("--normal-max-nn", type=int, default=30)
    ap.add_argument("--pca-axis", default="z", choices=("x", "y", "z"))
    ap.add_argument("--train-ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    models_dir = os.path.join(os.path.abspath(args.itodd_root), args.models_dir)
    out_root = os.path.abspath(args.out_root)

    model_files = sorted(f for f in os.listdir(models_dir) if f.startswith("obj_") and f.endswith(".ply"))
    if not model_files:
        raise RuntimeError(f"no models in {models_dir}")

    for sp in ("train", "test"):
        for sub in ("input", "gt", "obs", "meta"):
            os.makedirs(os.path.join(out_root, sp, sub), exist_ok=True)

    total = {"train": 0, "test": 0}

    for mf in model_files:
        model_path = os.path.join(models_dir, mf)
        obj_id = mf[len("obj_"):-len(".ply")]
        mesh = o3d.io.read_triangle_mesh(model_path)
        if mesh.is_empty():
            print(f"[SKIP] empty mesh: {mf}")
            continue

        for aug_id in range(args.num_aug_per_model):
            # (1) dynamic per-aug sampling
            pcd = mesh.sample_points_uniformly(number_of_points=args.num_cad_sample)
            gt_full = np.asarray(pcd.points, dtype=np.float32)
            gt_full, _ = normalize_by_bbox(gt_full)

            # (2) global SO(3)
            R_rand = Rotation.random().as_matrix().astype(np.float32)
            gt_rot = (gt_full @ R_rand.T).astype(np.float32)
            n_rot = estimate_normals(gt_rot, radius=args.normal_radius, max_nn=args.normal_max_nn)

            # (3) Fibonacci cameras (per-aug jitter -> different every time)
            cameras = fibonacci_sphere_cameras(args.num_views, args.hpr_sphere_r, jitter=args.fib_jitter)

            # split at (obj, aug) level
            split_name = _assign_split(f"obj{obj_id}_aug{aug_id}", args.train_ratio, args.seed)

            for vi in range(args.num_views):
                camera_pos = cameras[vi]

                # HPR
                pcd_hpr = o3d.geometry.PointCloud()
                pcd_hpr.points = o3d.utility.Vector3dVector(gt_rot.astype(np.float64))
                _, pt_map = pcd_hpr.hidden_point_removal(camera_pos.astype(np.float64), args.hpr_radius)
                pt_map = np.asarray(pt_map, dtype=np.int64)
                vis_pts = gt_rot[pt_map]
                vis_nrm = n_rot[pt_map] if n_rot.shape[0] == gt_rot.shape[0] else estimate_normals(vis_pts, args.normal_radius, args.normal_max_nn)

                # dropout
                mr = float(np.random.uniform(args.missing_rate_min, args.missing_rate_max))
                mr = min(mr, max(0.0, 1.0 - args.min_keep / max(vis_pts.shape[0], 1)))
                light_dir = None
                if np.random.rand() > 0.5:
                    cor, cor_n = apply_depth_dropout(vis_pts, vis_nrm, camera_pos=camera_pos, missing_rate=mr)
                else:
                    theta_l, phi_l = np.random.uniform(0, 2 * np.pi), np.random.uniform(0, np.pi)
                    light_dir = np.array([np.sin(phi_l) * np.cos(theta_l), np.sin(phi_l) * np.sin(theta_l), np.cos(phi_l)], dtype=np.float32)
                    cor, cor_n = apply_specular_dropout(vis_pts, vis_nrm, camera_pos=camera_pos, light_dir=light_dir, missing_rate=mr)

                cor, cor_n = _ensure_min_points(cor, cor_n, args.min_keep)

                # density-weighted resample -> 2048
                p_obs = density_weighted_resample(cor, camera_pos, args.num_obs, alpha=args.density_alpha)

                # bbox + pca
                p_norm, C_bbox = normalize_by_bbox(p_obs)
                p_in, R_pca, mu_pca = pca_align(p_norm, target_axis=args.pca_axis)
                gt_sync = (gt_rot - C_bbox[None, :]).astype(np.float32)
                gt_out = apply_pca_rigid(gt_sync, R_pca, mu_pca)
                gt_2048 = resample(gt_out, args.num_gt)

                stem = f"obj{obj_id}_aug{aug_id:04d}_v{vi}"
                sr = os.path.join(out_root, split_name)
                np.save(os.path.join(sr, "obs", f"{stem}.npy"), p_obs.astype(np.float32))
                np.save(os.path.join(sr, "input", f"{stem}.npy"), p_in.astype(np.float32))
                np.save(os.path.join(sr, "gt", f"{stem}.npy"), gt_2048.astype(np.float32))
                np.savez(
                    os.path.join(sr, "meta", f"{stem}.npz"),
                    C_bbox=C_bbox, R_pca=R_pca, mu_pca=mu_pca,
                    R_rand=R_rand, camera_pos=camera_pos,
                    missing_rate=np.float32(mr), n_visible=np.int32(vis_pts.shape[0]),
                    n_after_removal=np.int32(cor.shape[0]), view_idx=np.int32(vi),
                    aug_id=np.int32(aug_id), obj_id=np.array([obj_id]),
                )
                total[split_name] += 1

        print(f"[done] obj{obj_id}: {args.num_aug_per_model} aug × {args.num_views} views")

    print(f"\nDone. train={total['train']} test={total['test']} -> {out_root}")


if __name__ == "__main__":
    main()
