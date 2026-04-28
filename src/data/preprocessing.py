"""
PCN/ModelNet 共享：AABB 归一化、PCA 对齐、刚体、随机远距变换 T_far。
行向量约定: p' = p @ R.T + t

新管线（推荐）：``normalize_by_complete`` + 可选 ``random_gravity_axis_rot``，
不做 PCA 重定向；canonical input/gt 与 PCN 预训练分布一致，``obs_w`` 仍带 T_far。
"""
from __future__ import annotations

import numpy as np
import open3d as o3d


def normalize_by_complete(
    complete_obj: np.ndarray, partial_obj: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """以 complete 的 AABB 中心 + max-radius 单位球归一化，partial 跟随相同 (c, scale)。

    返回 (partial_cano, complete_cano, c, scale)，c 形状 (3,)，scale 标量。
    与 PCN 预训练分布一致：complete 落在以原点为心的单位球内。
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(complete_obj.astype(np.float64))
    c = np.asarray(pcd.get_axis_aligned_bounding_box().get_center(), dtype=np.float32)
    centered = (complete_obj - c[None, :]).astype(np.float32)
    scale = float(max(np.linalg.norm(centered, axis=1).max(), 1e-8))
    complete_cano = (centered / scale).astype(np.float32)
    partial_cano = ((partial_obj.astype(np.float32) - c[None, :]) / np.float32(scale)).astype(np.float32)
    return partial_cano, complete_cano, c, scale


def random_gravity_axis_rot(
    rng: np.random.Generator, max_deg: float, axis: str = "z"
) -> np.ndarray:
    """绕重力轴的小角度随机 SO(3) 旋转矩阵 (3x3, float32)。

    angle ~ Uniform[-max_deg, +max_deg]（度）。``axis`` ∈ {x,y,z}。
    """
    if max_deg <= 0.0:
        return np.eye(3, dtype=np.float32)
    deg = float(rng.uniform(-float(max_deg), float(max_deg)))
    a = np.deg2rad(deg)
    ca, sa = float(np.cos(a)), float(np.sin(a))
    ax = axis.lower()
    if ax == "x":
        m = np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=np.float64)
    elif ax == "y":
        m = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=np.float64)
    else:
        m = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]], dtype=np.float64)
    return m.astype(np.float32)


def normalize_by_bbox(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Center + scale: AABB center, then max distance = 1."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    c = np.asarray(pcd.get_axis_aligned_bounding_box().get_center(), dtype=np.float32)
    centered = (points - c[None, :]).astype(np.float32)
    scale = float(max(np.linalg.norm(centered, axis=1).max(), 1e-8))
    return (centered / scale).astype(np.float32), c, scale


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


def pca_align(
    points: np.ndarray, target_axis: str = "z", min_ratio: float = 1e-4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return pts.astype(np.float32), np.eye(3, dtype=np.float32), mu.astype(np.float32)

    v_main = evecs[:, order[0]].astype(np.float64)
    v_main /= np.linalg.norm(v_main) + 1e-12
    target_map = {"x": [1, 0, 0], "y": [0, 1, 0], "z": [0, 0, 1]}
    target = np.array(target_map[target_axis.lower()], dtype=np.float64)
    if np.dot(v_main, target) < 0:
        v_main = -v_main

    b = _orthonormal_frame(v_main)
    t = _orthonormal_frame(target)
    r = t @ b.T
    if np.linalg.det(r) < 0:
        t[:, 1] *= -1.0
        r = t @ b.T
    aligned = ((r @ (pts - mu).T).T + mu).astype(np.float32)
    return aligned, r.astype(np.float32), mu.astype(np.float32)


def apply_pca_rigid(points: np.ndarray, r: np.ndarray, mu: np.ndarray) -> np.ndarray:
    pts = points.astype(np.float64)
    r64, mu64 = r.astype(np.float64), mu.astype(np.float64).reshape(1, 3)
    return ((r64 @ (pts - mu64).T).T + mu64).astype(np.float32)


def inverse_pca(
    points_aligned: np.ndarray, r_pca: np.ndarray, mu_pca: np.ndarray
) -> np.ndarray:
    r = np.asarray(r_pca, dtype=np.float64).reshape(3, 3)
    mu = np.asarray(mu_pca, dtype=np.float64).reshape(1, 3)
    x = np.asarray(points_aligned, dtype=np.float64) - mu
    return ((r.T @ x.T).T + mu).astype(np.float32)


def sample_random_far_transform(
    rng: np.random.Generator,
    t_min: float,
    t_max: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    a = rng.standard_normal((3, 3)).astype(np.float64)
    q, _r = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    r = q.astype(np.float32)
    v = rng.standard_normal(3)
    v = (v / (np.linalg.norm(v) + 1e-12)).astype(np.float64)
    mag = float(rng.uniform(float(t_min), float(t_max)))
    t = (v * mag).astype(np.float32)
    return r, t, mag


def apply_rigid_row(
    points: np.ndarray, r: np.ndarray, t: np.ndarray
) -> np.ndarray:
    p = points.astype(np.float32)
    rr = r.astype(np.float32).reshape(3, 3)
    tt = t.astype(np.float32).reshape(1, 3)
    return (p @ rr.T + tt).astype(np.float32)


def rigid_T_4x4(r: np.ndarray, t: np.ndarray) -> np.ndarray:
    t3 = np.asarray(t, dtype=np.float64).reshape(3)
    rr = np.asarray(r, dtype=np.float64).reshape(3, 3)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rr.T
    out[:3, 3] = t3
    return out


def resample_rng(points: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((n, 3), dtype=np.float32)
    if points.shape[0] >= n:
        idx = rng.choice(points.shape[0], n, replace=False)
    else:
        idx = rng.choice(points.shape[0], n, replace=True)
    return points[idx].astype(np.float32)


def apply_inverse_normalization(p_comp: np.ndarray, meta: dict) -> np.ndarray:
    """补全点云从 canonical input 空间逆变换到 obs_w 世界系。

    新 schema（推荐）：meta 含 ``C_cano`` / ``scale_cano`` / ``R_far`` / ``t_far``
    （可选 ``R_aug``），还原路径为
    ``p_obj = (R_aug.T @ p_cano) * scale_cano + C_cano`` →
    ``p_w = p_obj @ R_far.T + t_far``。

    旧 schema（兼容）：meta 含 ``C_bbox`` / ``scale`` / ``R_pca`` (+ ``mu_pca``)，
    保持原 PCA 还原行为，输出仍为 obs_w 系。
    """
    if "C_cano" in meta:
        c_cano = np.asarray(meta["C_cano"], dtype=np.float32).reshape(1, 3)
        scale_cano = float(meta["scale_cano"]) if "scale_cano" in meta else 1.0
        r_aug = meta.get("R_aug")
        if r_aug is None:
            r_aug = np.eye(3, dtype=np.float32)
        r_aug = np.asarray(r_aug, dtype=np.float64).reshape(3, 3)
        p64 = np.asarray(p_comp, dtype=np.float64)
        p_obj = (p64 @ r_aug) * np.float64(scale_cano) + c_cano.astype(np.float64)
        if "R_far" in meta and "t_far" in meta:
            r_far = np.asarray(meta["R_far"], dtype=np.float64).reshape(3, 3)
            t_far = np.asarray(meta["t_far"], dtype=np.float64).reshape(1, 3)
            p_w = p_obj @ r_far.T + t_far
        else:
            p_w = p_obj
        return p_w.astype(np.float32)

    c_bbox = meta["C_bbox"].astype(np.float32).reshape(1, 3)
    scale = float(meta["scale"]) if "scale" in meta else 1.0
    r_pca = meta["R_pca"].astype(np.float32).reshape(3, 3)
    mu_pca = meta.get("mu_pca", np.zeros(3, dtype=np.float32)).astype(np.float32).reshape(3)
    p_norm = inverse_pca(p_comp, r_pca, mu_pca)
    return (p_norm * scale + c_bbox).astype(np.float32)
