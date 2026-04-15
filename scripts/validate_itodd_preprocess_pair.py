"""
scripts/validate_itodd_preprocess_pair.py

验证 iTODD 预处理输出的 (obs, input, gt, meta) 是否彼此对齐、坐标变换是否一致。

支持两种视角：
1) normalized view：直接叠加可视化 input(红) vs gt(绿)（两者应在同一归一化坐标系）
2) camera view：用 meta 逆变换把 gt 从训练坐标系还原回相机坐标系，与 obs(蓝) 叠加

用法示例：
  python3 scripts/validate_itodd_preprocess_pair.py --root data/itodd_processed_with_removal_16k_min1024_split --split train --random
  python3 scripts/validate_itodd_preprocess_pair.py --root data/itodd_processed_with_removal_16k_min1024_split --split train --stem itodd_s000001_im000000_inst000000_obj000001
"""

from __future__ import annotations

import argparse
import os
import random
from typing import Tuple, Optional

import numpy as np
import open3d as o3d


def _load_npy(path: str) -> np.ndarray:
    arr = np.load(path)
    return np.asarray(arr, dtype=np.float32)


def _load_meta(path: str):
    z = np.load(path, allow_pickle=True)
    # required keys from preprocessing scripts
    C_bbox = np.asarray(z["C_bbox"], dtype=np.float32).reshape(3)
    R_pca = np.asarray(z["R_pca"], dtype=np.float32).reshape(3, 3)
    mu_pca = np.asarray(z["mu_pca"], dtype=np.float32).reshape(3)
    return C_bbox, R_pca, mu_pca


def invert_pca_rigid(points_aligned: np.ndarray, R: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Forward in preprocessing: aligned = R @ (pts - mu) + mu
    Inverse: pts = R.T @ (aligned - mu) + mu
    """
    pts = np.asarray(points_aligned, dtype=np.float32)
    Rm = np.asarray(R, dtype=np.float32).reshape(3, 3)
    mu = np.asarray(mu, dtype=np.float32).reshape(1, 3)
    return ((pts - mu) @ Rm + mu).astype(np.float32)  # (aligned-mu) @ R  ==  R.T @ (aligned-mu)


def to_o3d(points: np.ndarray, color: Tuple[float, float, float]) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    pcd.paint_uniform_color(list(color))
    return pcd


def draw_clouds(clouds, window_name: str):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)
    for c in clouds:
        vis.add_geometry(c)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.05, 0.05, 0.08])
    vis.run()
    vis.destroy_window()


def find_sample(root: str, split: str, stem: Optional[str], random_pick: bool) -> str:
    input_dir = os.path.join(root, split, "input")
    if stem:
        path = os.path.join(input_dir, f"{stem}.npy")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        return stem

    files = [f for f in os.listdir(input_dir) if f.endswith(".npy")]
    if not files:
        raise RuntimeError(f"no npy found under {input_dir}")
    files.sort()
    if random_pick:
        f = random.choice(files)
    else:
        f = files[0]
    return os.path.splitext(f)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="处理后的 iTODD root（包含 train/test 子目录）")
    ap.add_argument("--split", default="train", choices=("train", "test"))
    ap.add_argument("--stem", default=None, help="指定样本 stem（不含扩展名），否则从 input 目录挑选")
    ap.add_argument("--random", action="store_true", help="不指定 stem 时随机选取样本")
    ap.add_argument("--no_normalized_view", action="store_true", help="不显示 normalized(input vs gt) 视图")
    ap.add_argument("--no_camera_view", action="store_true", help="不显示 camera(obs vs inv(gt)) 视图")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    stem = find_sample(root, args.split, args.stem, args.random)

    p_obs = os.path.join(root, args.split, "obs", f"{stem}.npy")
    p_in = os.path.join(root, args.split, "input", f"{stem}.npy")
    p_gt = os.path.join(root, args.split, "gt", f"{stem}.npy")
    p_meta = os.path.join(root, args.split, "meta", f"{stem}.npz")

    for p in (p_obs, p_in, p_gt, p_meta):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    obs = _load_npy(p_obs)
    inp = _load_npy(p_in)
    gt = _load_npy(p_gt)
    C_bbox, R_pca, mu_pca = _load_meta(p_meta)

    print(f"[sample] {stem}")
    print(f"[shapes] obs={obs.shape} input={inp.shape} gt={gt.shape}")
    print(f"[meta] ||C_bbox||={float(np.linalg.norm(C_bbox)):.3f}")

    if not args.no_normalized_view:
        # input 和 gt 应该在同一归一化坐标系
        clouds = [
            to_o3d(inp, (1.0, 0.0, 0.0)),   # red
            to_o3d(gt, (0.25, 0.85, 0.35)), # green
        ]
        draw_clouds(clouds, window_name="normalized: input(red) vs gt(green)")

    if not args.no_camera_view:
        # gt 在训练坐标系，逆 PCA 回 gt_sync，再 +C_bbox 回相机系
        gt_sync = invert_pca_rigid(gt, R_pca, mu_pca)
        gt_cam_recon = (gt_sync + C_bbox.reshape(1, 3)).astype(np.float32)

        clouds = [
            to_o3d(obs, (0.15, 0.45, 1.0)),       # blue
            to_o3d(gt_cam_recon, (0.25, 0.85, 0.35)),  # green
        ]
        draw_clouds(clouds, window_name="camera: obs(blue) vs recon_gt_cam(green)")


if __name__ == "__main__":
    main()

