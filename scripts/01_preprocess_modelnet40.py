"""
scripts/01_preprocess_modelnet40.py

ModelNet40 (.off) completion 训练集生成器。

增强管线（与 ITODD 统一）：
  1) 引入 aug 循环：每个模型生成 num_aug 份样本（类别反向加权平衡）
  2) 动态微观采样：每次 aug 重新从 mesh 采样 16384 点 + 全局 SO(3) 旋转
  3) Fibonacci 球面视角覆盖：每次 aug 生成 num_views 个相机（带 jitter）
  4) HPR 2.5D 可见性 → depth/specular dropout → 下限保护
  5) 非均匀深度密度重采样（近密远疏）→ 2048
  6) bbox 中心化 + PCA 对齐；GT 同步同一变换 → 2048

类别反向加权：统计各类别模型数，类别越少 → num_aug 越大，使最终样本量趋于均衡。
train/test 划分沿用 ModelNet40 原始目录结构（train/ 或 test/ 子目录）。
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections import Counter, defaultdict
from typing import List, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.transforms import apply_depth_dropout, apply_specular_dropout


# ═══════════════════════════ shared utilities ═══════════════════════════
# (identical to 00_preprocess_itodd.py for pipeline consistency)


def fibonacci_sphere_cameras(n: int, r: float, jitter: float = 0.0) -> np.ndarray:
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


def estimate_normals(points: np.ndarray, radius: float = 0.1, max_nn: int = 30) -> np.ndarray:
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


def _normalize_point_cloud(pcd: o3d.geometry.PointCloud):
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.translate(-pcd.get_center())
    pts = np.asarray(pcd.points, dtype=np.float32)
    nrm = np.asarray(pcd.normals, dtype=np.float32)
    scale = max(np.linalg.norm(pts, axis=1).max(), 1e-8)
    return (pts / scale).astype(np.float32), nrm


def _read_off_mesh(file_path: str) -> o3d.geometry.TriangleMesh:
    """Read .off with optional header repair."""
    with open(file_path, "rb") as f:
        raw = f.read().decode("utf-8", errors="ignore")
    first_line = raw.split("\n", 1)[0].strip()
    content = raw
    if first_line.startswith("OFF") and len(first_line) > 3 and first_line[3].isdigit():
        content = "OFF\n" + raw[3:].lstrip()
    _, tmp = tempfile.mkstemp(suffix=".off")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
        mesh = o3d.io.read_triangle_mesh(tmp)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return mesh


# ═══════════════════════════ scanning & class balancing ═══════════════════════════


def _scan_dataset(raw_dir: str):
    """
    Return list of (file_path, category, split) and per-split category counts.
    """
    entries = []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if not f.endswith(".off"):
                continue
            fp = os.path.join(root, f)
            parts = fp.replace("\\", "/").split("/")
            try:
                idx = parts.index("ModelNet40")
                cat = parts[idx + 1]
                split = parts[idx + 2]
                entries.append((fp, cat, split))
            except (ValueError, IndexError):
                continue
    return entries


def _compute_aug_counts(entries, base_aug: int, max_aug: int):
    """
    Inverse-frequency class balancing per split.
    Returns dict[(cat, split)] -> num_aug for that category in that split.
    """
    split_cat_counts: dict[str, Counter] = defaultdict(Counter)
    for _, cat, split in entries:
        split_cat_counts[split][cat] += 1

    aug_map = {}
    for split, counts in split_cat_counts.items():
        max_count = max(counts.values())
        for cat, cnt in counts.items():
            ratio = max_count / max(cnt, 1)
            aug = min(max(int(round(base_aug * ratio)), 1), max_aug)
            aug_map[(cat, split)] = aug
    return aug_map


# ═══════════════════════════ main ═══════════════════════════


def main():
    ap = argparse.ArgumentParser(description="ModelNet40 completion 预处理（类别平衡 + 统一增强）")
    ap.add_argument("--raw-dir", default="/home/csy/SnowflakeNet_FPFH_ICP/data/raw/ModelNet40")
    ap.add_argument("--out-root", default="/home/csy/SnowflakeNet_FPFH_ICP/data/processed/modelnet40")
    ap.add_argument("--base-aug", type=int, default=1, help="最大类别对应的 aug 次数")
    ap.add_argument("--max-aug", type=int, default=10, help="反向加权后单模型最大 aug 上限")
    ap.add_argument("--num-views", type=int, default=8)
    ap.add_argument("--num-cad-sample", type=int, default=16384)
    ap.add_argument("--num-obs", type=int, default=2048)
    ap.add_argument("--num-gt", type=int, default=2048)
    ap.add_argument("--min-keep", type=int, default=1024)
    ap.add_argument("--missing-rate-min", type=float, default=0.1)
    ap.add_argument("--missing-rate-max", type=float, default=0.4)
    ap.add_argument("--hpr-sphere-r", type=float, default=3.0)
    ap.add_argument("--hpr-radius", type=float, default=100.0)
    ap.add_argument("--fib-jitter", type=float, default=0.15)
    ap.add_argument("--density-alpha", type=float, default=1.5)
    ap.add_argument("--normal-radius", type=float, default=0.1)
    ap.add_argument("--normal-max-nn", type=int, default=30)
    ap.add_argument("--pca-axis", default="z", choices=("x", "y", "z"))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    np.random.seed(args.seed)
    out_root = os.path.abspath(args.out_root)

    if not os.path.exists(args.raw_dir):
        print(f"原始数据集路径不存在: {args.raw_dir}")
        return

    entries = _scan_dataset(args.raw_dir)
    if not entries:
        print("未找到任何 .off 文件")
        return

    aug_map = _compute_aug_counts(entries, args.base_aug, args.max_aug)

    # show balance info
    splits_cats = defaultdict(set)
    for _, cat, split in entries:
        splits_cats[split].add(cat)
    for split in sorted(splits_cats):
        cats = sorted(splits_cats[split])
        print(f"[{split}] {len(cats)} classes, aug samples:")
        for cat in cats[:5]:
            print(f"  {cat}: aug={aug_map.get((cat, split), 1)}")
        if len(cats) > 5:
            print(f"  ... ({len(cats)} total)")

    for sp in ("train", "test"):
        for sub in ("input", "gt", "obs", "meta"):
            os.makedirs(os.path.join(out_root, sp, sub), exist_ok=True)

    total = {"train": 0, "test": 0}

    for file_path, cat, split in tqdm(entries, desc="Processing"):
        mesh = _read_off_mesh(file_path)
        if not mesh.has_triangles():
            continue

        base_name = os.path.basename(file_path).replace(".off", "")
        num_aug = aug_map.get((cat, split), 1)

        for aug_id in range(num_aug):
            # (1) dynamic per-aug sampling + normalize
            pcd = mesh.sample_points_uniformly(number_of_points=args.num_cad_sample)
            gt_full, gt_nrm = _normalize_point_cloud(pcd)

            # (2) global SO(3)
            R_rand = Rotation.random().as_matrix().astype(np.float32)
            gt_rot = (gt_full @ R_rand.T).astype(np.float32)
            n_rot = (gt_nrm @ R_rand.T).astype(np.float32)

            gt_2048_base = resample(gt_rot, args.num_gt)

            # (3) Fibonacci cameras
            cameras = fibonacci_sphere_cameras(args.num_views, args.hpr_sphere_r, jitter=args.fib_jitter)

            for vi in range(args.num_views):
                camera_pos = cameras[vi]

                # HPR
                pcd_hpr = o3d.geometry.PointCloud()
                pcd_hpr.points = o3d.utility.Vector3dVector(gt_rot.astype(np.float64))
                _, pt_map = pcd_hpr.hidden_point_removal(camera_pos.astype(np.float64), args.hpr_radius)
                pt_map = np.asarray(pt_map, dtype=np.int64)
                vis_pts = gt_rot[pt_map]
                vis_nrm = n_rot[pt_map]

                # dropout
                mr = float(np.random.uniform(args.missing_rate_min, args.missing_rate_max))
                mr = min(mr, max(0.0, 1.0 - args.min_keep / max(vis_pts.shape[0], 1)))
                if np.random.rand() > 0.5:
                    cor, cor_n = apply_depth_dropout(vis_pts, vis_nrm, camera_pos=camera_pos, missing_rate=mr)
                else:
                    theta_l, phi_l = np.random.uniform(0, 2 * np.pi), np.random.uniform(0, np.pi)
                    ld = np.array([np.sin(phi_l) * np.cos(theta_l), np.sin(phi_l) * np.sin(theta_l), np.cos(phi_l)], dtype=np.float32)
                    cor, cor_n = apply_specular_dropout(vis_pts, vis_nrm, camera_pos=camera_pos, light_dir=ld, missing_rate=mr)

                cor, cor_n = _ensure_min_points(cor, cor_n, args.min_keep)

                # density-weighted resample
                p_obs = density_weighted_resample(cor, camera_pos, args.num_obs, alpha=args.density_alpha)

                # bbox + pca
                p_norm, C_bbox = normalize_by_bbox(p_obs)
                p_in, R_pca, mu_pca = pca_align(p_norm, target_axis=args.pca_axis)
                gt_sync = (gt_rot - C_bbox[None, :]).astype(np.float32)
                gt_out = apply_pca_rigid(gt_sync, R_pca, mu_pca)
                gt_2048 = resample(gt_out, args.num_gt)

                if num_aug > 1:
                    stem = f"{base_name}_aug{aug_id:02d}_v{vi}"
                else:
                    stem = f"{base_name}_v{vi}"

                sr = os.path.join(out_root, split)
                np.save(os.path.join(sr, "obs", f"{stem}.npy"), p_obs.astype(np.float32))
                np.save(os.path.join(sr, "input", f"{stem}.npy"), p_in.astype(np.float32))
                np.save(os.path.join(sr, "gt", f"{stem}.npy"), gt_2048.astype(np.float32))
                np.savez(
                    os.path.join(sr, "meta", f"{stem}.npz"),
                    C_bbox=C_bbox, R_pca=R_pca, mu_pca=mu_pca,
                    R_rand=R_rand, camera_pos=camera_pos,
                    missing_rate=np.float32(mr), n_visible=np.int32(vis_pts.shape[0]),
                    n_after_removal=np.int32(cor.shape[0]), view_idx=np.int32(vi),
                    aug_id=np.int32(aug_id), category=np.array([cat]),
                )
                total[split] += 1

    print(f"\nDone. train={total['train']} test={total['test']} -> {out_root}")


if __name__ == "__main__":
    main()
