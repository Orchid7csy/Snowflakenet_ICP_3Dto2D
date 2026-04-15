"""
scripts/00_preprocess_itodd_with_removal_16k_min1024.py

把 ITODD(BOP) 的真实 RGB-D 观测转为 completion 微调所需点云对，并“按同样逻辑”加入：

- 对 obs 额外施加 depth/specular dropout（来自 src/data/transforms.py）
- dropout 前动态上限化 missing_rate，保证剔除后点数 >= min_keep_after_removal（默认 1025，即 >1024）
- CAD 采样密度默认提高到 16384，降低 gt 的离散化误差

注意：
- ITODD 的 obs 本身就是单视角深度反投影点云，不再额外做 HPR（等价于已经是可见点集）。
- dropout 的 camera_pos 使用相机坐标系原点 [0,0,0]（点云也在相机坐标系）。
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Tuple

import cv2
import numpy as np
import open3d as o3d

# allow running as: python3 scripts/xxx.py from repo root
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.transforms import apply_depth_dropout, apply_specular_dropout


def load_K(camera_json: str) -> Tuple[np.ndarray, int, int]:
    with open(camera_json, "r", encoding="utf-8") as f:
        cam = json.load(f)
    fx, fy, cx, cy = float(cam["fx"]), float(cam["fy"]), float(cam["cx"]), float(cam["cy"])
    w, h = int(cam["width"]), int(cam["height"])
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
    return K, w, h


def backproject_depth_mask(depth_tif: str, mask_png: str, K: np.ndarray) -> np.ndarray:
    depth = cv2.imread(depth_tif, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(depth_tif)
    mask = cv2.imread(mask_png, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(mask_png)

    Z = depth.astype(np.float32)
    m = (mask > 0) & (Z > 0)
    vs, us = np.where(m)
    if us.size == 0:
        return np.zeros((0, 3), dtype=np.float32)

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    z = Z[vs, us]
    x = (us.astype(np.float32) - cx) / fx * z
    y = (vs.astype(np.float32) - cy) / fy * z
    return np.stack([x, y, z], axis=1).astype(np.float32)


def sample_cad_points(model_ply: str, n: int) -> np.ndarray:
    mesh = o3d.io.read_triangle_mesh(model_ply)
    if mesh.is_empty():
        raise RuntimeError(f"bad mesh: {model_ply}")
    pcd = mesh.sample_points_uniformly(number_of_points=n)
    return np.asarray(pcd.points, dtype=np.float32)


def apply_T(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return (points @ R.T + t[None, :]).astype(np.float32)


def normalize_by_bbox(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    bbox = pcd.get_axis_aligned_bounding_box()
    c = np.asarray(bbox.get_center(), dtype=np.float32)
    return (points - c).astype(np.float32), c


def _orthonormal_frame_from_axis(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    v = v / n
    tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(v, tmp)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
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
    lam1 = float(evals[order[0]])
    lam2 = float(evals[order[1]]) if len(order) > 1 else 0.0
    if lam1 < 1e-12 or (lam1 - lam2) / (lam1 + 1e-12) < min_ratio:
        return points.astype(np.float32), np.eye(3, dtype=np.float32), mu.astype(np.float32)

    v_main = evecs[:, order[0]].astype(np.float64)
    v_main /= np.linalg.norm(v_main) + 1e-12

    ax = target_axis.lower()
    target = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if ax == "x":
        target = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    elif ax == "y":
        target = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    if np.dot(v_main, target) < 0:
        v_main = -v_main

    B = _orthonormal_frame_from_axis(v_main)
    T = _orthonormal_frame_from_axis(target)
    R = T @ B.T
    if np.linalg.det(R) < 0:
        T[:, 1] *= -1.0
        R = T @ B.T

    aligned = ((R @ (pts - mu).T).T + mu).astype(np.float32)
    return aligned, R.astype(np.float32), mu.astype(np.float32)


def apply_pca_rigid(points: np.ndarray, R: np.ndarray, mu: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float64)
    R64 = R.astype(np.float64)
    mu64 = mu.astype(np.float64).reshape(1, 3)
    return ((R64 @ (pts - mu64).T).T + mu64).astype(np.float32)


def resample(points: np.ndarray, n: int) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((n, 3), dtype=np.float32)
    if points.shape[0] >= n:
        idx = np.random.choice(points.shape[0], n, replace=False)
    else:
        idx = np.random.choice(points.shape[0], n, replace=True)
    return points[idx].astype(np.float32)


def estimate_normals(points: np.ndarray, radius: float = 20.0, max_nn: int = 30) -> np.ndarray:
    """
    ITODD 的坐标是 mm（深度 tif 通常是 uint16 mm），所以这里 radius 用 mm 量级。
    """
    if points.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=float(radius), max_nn=int(max_nn)))
    return np.asarray(pcd.normals, dtype=np.float32)


def _cap_missing_rate_for_min_keep(n_visible: int, expected_rate: float, min_keep: int) -> float:
    if n_visible <= 0:
        return 0.0
    if n_visible <= min_keep:
        return 0.0
    r_max = 1.0 - (float(min_keep) / float(n_visible))
    return float(np.clip(min(expected_rate, r_max), 0.0, 0.95))


def _ensure_min_points(points: np.ndarray, normals: np.ndarray, min_keep: int):
    n = int(points.shape[0])
    if n >= min_keep or n == 0:
        return points, normals
    idx = np.random.choice(n, min_keep - n, replace=True)
    pts = np.concatenate([points, points[idx]], axis=0)
    nrm = np.concatenate([normals, normals[idx]], axis=0)
    return pts.astype(np.float32), nrm.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--itodd-root", default="/home/csy/SnowflakeNet_FPFH_ICP/ITODD")
    ap.add_argument("--scene", default="000001", help="scene folder under val/val/")
    ap.add_argument("--split", default="train", choices=("train", "test"))
    ap.add_argument("--out-root", default="/home/csy/SnowflakeNet_FPFH_ICP/data/itodd_processed_with_removal_16k_min1024")
    ap.add_argument("--camera-json", default="base/itoddmv/camera_3dlong.json")
    ap.add_argument("--scene-gt", default="val/val/000001/scene_gt_3dlong.json")
    ap.add_argument("--scene-gt-info", default="val/val/000001/scene_gt_info_3dlong.json")
    ap.add_argument("--depth-dir", default="val/val/000001/depth_3dlong")
    ap.add_argument("--mask-dir", default="val/val/000001/mask_visib_3dlong")
    ap.add_argument("--models-dir", default="models/models")
    ap.add_argument("--num-obs", type=int, default=2048)
    ap.add_argument("--num-gt", type=int, default=2048)
    ap.add_argument("--num-cad-sample", type=int, default=16384)
    ap.add_argument("--pca-axis", default="z", choices=("x", "y", "z"))
    ap.add_argument("--expected-missing-rate", type=float, default=0.2)
    ap.add_argument("--min-keep-after-removal", type=int, default=1025)
    ap.add_argument("--normal-radius-mm", type=float, default=20.0)
    ap.add_argument("--normal-max-nn", type=int, default=30)
    args = ap.parse_args()

    root = os.path.abspath(args.itodd_root)
    out_root = os.path.abspath(args.out_root)
    split_root = os.path.join(out_root, args.split)
    for sub in ("input", "gt", "obs", "meta"):
        os.makedirs(os.path.join(split_root, sub), exist_ok=True)

    K, _, _ = load_K(os.path.join(root, args.camera_json))

    with open(os.path.join(root, args.scene_gt), "r", encoding="utf-8") as f:
        scene_gt = json.load(f)
    with open(os.path.join(root, args.scene_gt_info), "r", encoding="utf-8") as f:
        scene_gt_info = json.load(f)

    depth_dir = os.path.join(root, args.depth_dir)
    mask_dir = os.path.join(root, args.mask_dir)
    models_dir = os.path.join(root, args.models_dir)

    depth_files = sorted([f for f in os.listdir(depth_dir) if f.endswith(".tif")])
    total = 0
    for df in depth_files:
        im_id = int(os.path.splitext(df)[0])
        if str(im_id) not in scene_gt:
            continue
        anns = scene_gt[str(im_id)]
        infos = scene_gt_info.get(str(im_id), [])

        for inst_id, ann in enumerate(anns):
            obj_id = int(ann["obj_id"])
            mask_path = os.path.join(mask_dir, f"{im_id:06d}_{inst_id:06d}.png")
            if not os.path.exists(mask_path):
                continue

            depth_path = os.path.join(depth_dir, f"{im_id:06d}.tif")
            obs_full = backproject_depth_mask(depth_path, mask_path, K)
            if obs_full.shape[0] < 50:
                continue

            # GT from CAD in camera frame
            model_path = os.path.join(models_dir, f"obj_{obj_id:06d}.ply")
            if not os.path.exists(model_path):
                continue
            cad = sample_cad_points(model_path, int(args.num_cad_sample))
            R = np.array(ann["cam_R_m2c"], dtype=np.float32).reshape(3, 3)
            t = np.array(ann["cam_t_m2c"], dtype=np.float32).reshape(3)
            gt_cam = apply_T(cad, R, t)

            # 先把 obs 重采样到 2048，再在同一尺度上做 dropout（对应 ModelNet 脚本里“先得到 corrupted 再采样 2048”的作用）
            obs_rs = resample(obs_full, int(args.num_obs))
            obs_normals = estimate_normals(
                obs_rs, radius=float(args.normal_radius_mm), max_nn=int(args.normal_max_nn),
            )

            # 在真实单视角点云上额外模拟缺失：camera_pos 取相机坐标系原点
            camera_pos = np.zeros(3, dtype=np.float32)
            missing_rate = _cap_missing_rate_for_min_keep(
                n_visible=int(obs_rs.shape[0]),
                expected_rate=float(args.expected_missing_rate),
                min_keep=int(args.min_keep_after_removal),
            )

            if np.random.rand() > 0.5:
                obs_cor, obs_cor_n = apply_depth_dropout(
                    obs_rs, obs_normals, camera_pos=camera_pos, missing_rate=missing_rate,
                )
            else:
                theta_l = np.random.uniform(0, 2 * np.pi)
                phi_l = np.random.uniform(0, np.pi)
                light_dir = np.array(
                    [np.sin(phi_l) * np.cos(theta_l), np.sin(phi_l) * np.sin(theta_l), np.cos(phi_l)],
                    dtype=np.float32,
                )
                obs_cor, obs_cor_n = apply_specular_dropout(
                    obs_rs, obs_normals, camera_pos=camera_pos, light_dir=light_dir, missing_rate=missing_rate,
                )

            obs_cor, obs_cor_n = _ensure_min_points(obs_cor, obs_cor_n, int(args.min_keep_after_removal))

            # 再采样为网络输入 2048（和你其他数据管线对齐）
            obs_out = resample(obs_cor, int(args.num_obs))

            # 用 obs bbox center 做归一化中心（不使用 mean）
            p_norm, C_bbox = normalize_by_bbox(obs_out)
            gt_sync = (gt_cam - C_bbox[None, :]).astype(np.float32)

            # PCA align based on normalized obs
            p_in, R_pca, mu_pca = pca_align(p_norm, target_axis=args.pca_axis)
            gt_out = apply_pca_rigid(gt_sync, R_pca, mu_pca)
            gt_out = resample(gt_out, int(args.num_gt))

            stem = f"itodd_s{args.scene}_im{im_id:06d}_inst{inst_id:06d}_obj{obj_id:06d}"
            np.save(os.path.join(split_root, "obs", f"{stem}.npy"), obs_out.astype(np.float32))
            np.save(os.path.join(split_root, "input", f"{stem}.npy"), p_in.astype(np.float32))
            np.save(os.path.join(split_root, "gt", f"{stem}.npy"), gt_out.astype(np.float32))
            np.savez(
                os.path.join(split_root, "meta", f"{stem}.npz"),
                R_rand=np.eye(3, dtype=np.float32),
                t_rand=np.zeros(3, dtype=np.float32),
                C_bbox=C_bbox.astype(np.float32),
                R_pca=R_pca.astype(np.float32),
                mu_pca=mu_pca.astype(np.float32),
                itodd_obj_id=np.array([obj_id], dtype=np.int32),
                itodd_im_id=np.array([im_id], dtype=np.int32),
                itodd_inst_id=np.array([inst_id], dtype=np.int32),
                expected_rate=np.asarray(float(args.expected_missing_rate), dtype=np.float32),
                missing_rate_used=np.asarray(float(missing_rate), dtype=np.float32),
                n_obs_before=np.asarray(int(obs_full.shape[0]), dtype=np.int32),
                n_obs_rs=np.asarray(int(obs_rs.shape[0]), dtype=np.int32),
                n_after_removal=np.asarray(int(obs_cor.shape[0]), dtype=np.int32),
                min_keep_after_removal=np.asarray(int(args.min_keep_after_removal), dtype=np.int32),
                num_cad_sample=np.asarray(int(args.num_cad_sample), dtype=np.int32),
            )
            total += 1

    print(f"Done. wrote {total} samples to {split_root}")


if __name__ == "__main__":
    main()

