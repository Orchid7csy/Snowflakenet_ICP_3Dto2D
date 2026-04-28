"""点云 I/O 与 Open3D 辅助。"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import open3d as o3d


def read_pcd_xyz(path: str | Path) -> np.ndarray:
    pc = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pc.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError(f"空点云: {path}")
    return pts.astype(np.float32)


def save_npy(path: str | Path, pts: np.ndarray) -> None:
    os.makedirs(os.path.dirname(str(path)) or ".", exist_ok=True)
    np.save(str(path), pts.astype(np.float32))


def load_meta_npz(path: str | Path) -> dict:
    return dict(np.load(str(path), allow_pickle=True))


def to_o3d_pcd(points: np.ndarray, color: tuple[float, float, float] | None = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if color is not None:
        pcd.paint_uniform_color(list(color))
    return pcd
