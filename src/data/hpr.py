"""Fibonacci 球面相机 + HPR 可见性（与 compute_pcn_best_hpr_view 一致）。"""
from __future__ import annotations

import numpy as np
import open3d as o3d


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


def hpr_radius_effective(
    hpr_sphere_r: float, hpr_radius: float, hpr_radius_factor: float
) -> float:
    return max(float(hpr_radius), float(hpr_radius_factor) * (float(hpr_sphere_r) + 1.0))


def max_hpr_visibility_count(
    gt_aligned: np.ndarray,
    *,
    num_views: int,
    hpr_sphere_r: float,
    hpr_radius_eff: float,
) -> tuple[int, int]:
    if gt_aligned.shape[0] == 0:
        return 0, 0
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(gt_aligned.astype(np.float64))
    cams = fibonacci_sphere_cameras(int(num_views), float(hpr_sphere_r), jitter=0.0)
    best_n, best_i = -1, 0
    for i in range(int(cams.shape[0])):
        cam = cams[i].astype(np.float64)
        _, pt_map = pcd.hidden_point_removal(cam, float(hpr_radius_eff))
        n = int(np.asarray(pt_map).size)
        if n > best_n:
            best_n, best_i = n, i
    return best_n, best_i
