"""
点云与刚体位姿的评估指标 (CPU, NumPy + SciPy; 可选与 Open3D/06 的 SE(3) 对齐)。

约定
----
- 齐次阵 **T (4x4, 列向量)** : ``p_new = T @ p_h``, ``p_h = [x,y,z,1]``.
- 预处理 **行向量** (01/14/06 的 ``p' = p @ R.T + t``) 与 T 的转换见 ``T_row_rigid_to_column_4x4``。
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

Array = np.ndarray


def T_row_rigid_to_column_4x4(
    R: Array,
    t: Array,
) -> np.ndarray:
    """
    行向量刚体: ``p' = p @ R^T + t`` (p 为 (N,3))  对应 4x4 列向量右乘.

    即 ``p_h' = T @ p_h``，其中 T[:3,:3] = R, T[:3,3] = t (列)
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def T_column_se3_to_Rt(T: Array) -> Tuple[np.ndarray, np.ndarray]:
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return T[:3, :3].copy(), T[:3, 3].copy()


def se3_to_R_t(T: Array) -> Tuple[np.ndarray, np.ndarray]:
    """Alias of ``T_column_se3_to_Rt`` (列向量 4x4 -> R, t)."""
    return T_column_se3_to_Rt(T)


def transform_points_T_column(p: Array, T: Array) -> np.ndarray:
    """p: (N,3), T: 4x4 列. 返回 (N,3)。"""
    p = np.asarray(p, dtype=np.float64).reshape(-1, 3)
    T = np.asarray(T, dtype=np.float64).reshape(4, 4)
    R, t = T[:3, :3], T[:3, 3]
    return (R @ p.T).T + t[None, :]


# ----- 点云: 最近邻辅助 -----


def _knn_dists(pts_a: Array, tree_b: cKDTree) -> np.ndarray:
    """pts_a: (N,3), 返回 pts_a 中每点到 tree_b 的最近距离."""
    pts_a = np.asarray(pts_a, dtype=np.float64).reshape(-1, 3)
    try:
        d, _ = tree_b.query(pts_a, k=1, workers=-1)
    except TypeError:
        d, _ = tree_b.query(pts_a, k=1)
    return np.asarray(d, dtype=np.float64)


# ----- Chamfer / F-Score -----


def _chamfer_l2_symmetric(p_a: Array, p_b: Array) -> float:
    """L2 型对称: 0.5 * (sqrt(mean d_ab^2) + sqrt(mean d_ba^2))，d 为欧氏最近距离."""
    a = np.asarray(p_a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(p_b, dtype=np.float64).reshape(-1, 3)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    tree_a, tree_b = cKDTree(a), cKDTree(b)
    d_ab = _knn_dists(b, tree_a) ** 2
    d_ba = _knn_dists(a, tree_b) ** 2
    return 0.5 * (float(np.sqrt(np.mean(d_ab))) + float(np.sqrt(np.mean(d_ba))))


def chamfer_distance(
    p_a: Array,
    p_b: Array,
    *,
    norm: int = 1,
) -> float:
    """
    对称 Chamfer. norm=1: 0.5 * (mean d_ab + mean d_ba)（欧距）.
    norm=2: 0.5 * (sqrt(mean d_ab^2) + sqrt(mean d_ba^2)).
    p_a, p_b: (N,3) / (M,3)
    """
    a = np.asarray(p_a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(p_b, dtype=np.float64).reshape(-1, 3)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return float("nan")
    if norm not in (1, 2):
        raise ValueError("norm 应为 1 或 2")
    if norm == 1:
        tree_a, tree_b = cKDTree(a), cKDTree(b)
        d_ab = _knn_dists(b, tree_a)
        d_ba = _knn_dists(a, tree_b)
        return 0.5 * (float(np.mean(d_ab)) + float(np.mean(d_ba)))
    return _chamfer_l2_symmetric(p_a, p_b)


def chamfer_distance_l1(p_a: Array, p_b: Array) -> float:
    """对称 Chamfer，L1 型 (最近欧氏距的均值对称)."""
    return chamfer_distance(p_a, p_b, norm=1)


def chamfer_distance_l2(p_a: Array, p_b: Array) -> float:
    """对称 Chamfer，L2 型 (见 :func:`chamfer_distance` 的 norm=2)."""
    return chamfer_distance(p_a, p_b, norm=2)


def f_score(
    p_a: Array,
    p_b: Array,
    *,
    threshold: float,
    beta: float = 1.0,
) -> Dict[str, float]:
    """
    F_beta: P = 满足 min_b|a-b|<d 的 a 占 A 比; R = 满足 min_a|a-b|<d 的 b 占 B 比;
    F = (1+b^2) P R / (b^2 P + R). threshold 与点云坐标单位一致.
    """
    a = np.asarray(p_a, dtype=np.float64).reshape(-1, 3)
    b = np.asarray(p_b, dtype=np.float64).reshape(-1, 3)
    if a.shape[0] == 0 or b.shape[0] == 0:
        return {"precision": 0.0, "recall": 0.0, "f": 0.0}
    tree_b, tree_a = cKDTree(b), cKDTree(a)
    d_ba = _knn_dists(a, tree_b)
    d_ab = _knn_dists(b, tree_a)
    p = float(np.mean(d_ba < threshold))
    r = float(np.mean(d_ab < threshold))
    b2 = beta * beta
    f = 0.0
    if p + 1e-12 > 0 and r + 1e-12 > 0:
        f = (1.0 + b2) * p * r / (b2 * p + r + 1e-12)
    return {"precision": p, "recall": r, "f": f}


# ----- 位姿 -----


def rotation_error_deg(R_pred: Array, R_gt: Array) -> float:
    """
    相对旋转角 (度). R_err = R_gt^T R_pred; 测地角在 SO(3) 上.
    """
    Rp = np.asarray(R_pred, dtype=np.float64).reshape(3, 3)
    Rg = np.asarray(R_gt, dtype=np.float64).reshape(3, 3)
    Rerr = Rg.T @ Rp
    rot = Rotation.from_matrix(Rerr)
    ang = float(np.linalg.norm(rot.as_rotvec()))
    return float(np.degrees(ang))


def translation_error_l2(
    t_pred: Array,
    t_gt: Array,
    *,
    to_cm: bool = False,
    mm_per_unit: Optional[float] = None,
) -> float:
    """
    平移 L2. 若 to_cm: 点云在「归一化单位」时，可先设 mm_per_unit(每单位=多少 mm) 再/10 为 cm; 或调用方
    先换物理单位。默认直接 ``||t_pred - t_gt||_2`` (与 06/ meta 的 t 同单位).
    """
    d = float(np.linalg.norm(np.asarray(t_pred) - np.asarray(t_gt), ord=2))
    if to_cm and mm_per_unit is not None:
        return d * float(mm_per_unit) / 10.0
    return d


def se3_error_deg_cm(
    T_pred: Array,
    T_gt: Array,
) -> Dict[str, float]:
    """
    从 4x4 齐次阵比较 R,t: rot_deg, trans_l2.
    T_pred, T_gt 须为**同一**约定 (列向量右乘).
    """
    Rp, tp = T_column_se3_to_Rt(T_pred)
    Rg, tg = T_column_se3_to_Rt(T_gt)
    return {
        "rot_deg": rotation_error_deg(Rp, Rg),
        "trans_l2": float(np.linalg.norm(tp - tg)),
    }


def add_s_distance(
    p_src: Array,
    p_tgt: Array,
    T_src_to_tgt: Array,
    *,
    symmetric: bool = False,
) -> float:
    """
    ADD-S: 对源点施加 T 后, mean_a min_b ||T a - b||. optional symmetric 再加反向 mean.
    """
    ps = transform_points_T_column(p_src, T_src_to_tgt)
    pt = np.asarray(p_tgt, dtype=np.float64).reshape(-1, 3)
    if ps.shape[0] == 0 or pt.shape[0] == 0:
        return float("nan")
    tree = cKDTree(pt)
    try:
        d, _ = tree.query(ps, k=1, workers=-1)
    except TypeError:
        d, _ = tree.query(ps, k=1)
    s1 = float(np.mean(d))
    if not symmetric:
        return s1
    tree_s = cKDTree(ps)
    try:
        d2, _ = tree_s.query(pt, k=1, workers=-1)
    except TypeError:
        d2, _ = tree_s.query(pt, k=1)
    s2 = float(np.mean(d2))
    return 0.5 * (s1 + s2)


def add_s(
    p_src: Array,
    p_tgt: Array,
    T_4x4: Array,
    *,
    symmetric: bool = False,
) -> float:
    """ADD-S: :func:`add_s_distance` 的简称（不推荐直接用于位姿评估，见 :func:`add_s_cad`）。"""
    return add_s_distance(p_src, p_tgt, T_4x4, symmetric=symmetric)


def add_s_cad(
    p_cad: Array,
    T_pred_cad2world: Array,
    T_gt_cad2world: Array,
    *,
    symmetric: bool = True,
) -> float:
    """
    **标准 ADD-S**（论文写法）。必须以**完整 CAD**（规范位姿 / origin frame）为 x：

        p_gt  = T_gt  * p_cad
        p_est = T_est * p_cad
        ADD-S = mean_i  min_j  || p_est[i] - p_gt[j] ||

    - 不再用残缺/obs 点云（避免 `T_icp≈I` 时自匹配导致 ADD-S≈0 的假象）。
    - 两个变换必须**同向**（都是 CAD→world 或都是 world→CAD），否则无意义。
    - ``symmetric=True``（默认）再取 ``p_gt -> p_est`` 方向的均值。
    """
    p_est = transform_points_T_column(p_cad, T_pred_cad2world)
    p_gt_tr = transform_points_T_column(p_cad, T_gt_cad2world)
    if p_est.shape[0] == 0 or p_gt_tr.shape[0] == 0:
        return float("nan")
    tree_gt = cKDTree(p_gt_tr)
    try:
        d, _ = tree_gt.query(p_est, k=1, workers=-1)
    except TypeError:
        d, _ = tree_gt.query(p_est, k=1)
    s1 = float(np.mean(d))
    if not symmetric:
        return s1
    tree_est = cKDTree(p_est)
    try:
        d2, _ = tree_est.query(p_gt_tr, k=1, workers=-1)
    except TypeError:
        d2, _ = tree_est.query(p_gt_tr, k=1)
    return 0.5 * (s1 + float(np.mean(d2)))


# ----- 兼容 plan 的别名 -----


def translation_error(
    t_pred: Array,
    t_gt: Array,
    *,
    unit: str = "same",
    scale_to_cm: Optional[float] = None,
) -> float:
    """
    若 unit=='cm' 且提供 scale_to_cm(每 1 个坐标单位 = 多少厘米), 则结果乘 scale_to_cm.
    否则为与输入相同单位的 L2.
    """
    if unit not in ("same", "cm"):
        raise ValueError("unit 应为 'same' 或 'cm'")
    d = float(np.linalg.norm(np.asarray(t_pred) - np.asarray(t_gt), ord=2))
    if unit == "cm" and scale_to_cm is not None:
        return d * float(scale_to_cm)
    return d


def rotation_error_deg_T(T_pred: Array, T_gt: Array) -> float:
    Rp, _ = T_column_se3_to_Rt(T_pred)
    Rg, _ = T_column_se3_to_Rt(T_gt)
    return rotation_error_deg(Rp, Rg)


if __name__ == "__main__":
    if hasattr(np.random, "default_rng"):
        _rng = np.random.default_rng(0)
        _p = _rng.random((20, 3))
    else:
        np.random.seed(0)
        _p = np.random.rand(20, 3).astype(np.float64)
    assert chamfer_distance_l1(_p, _p) < 1e-9
    R = Rotation.from_euler("z", np.deg2rad(3.0)).as_matrix()
    assert abs(rotation_error_deg(R, np.eye(3)) - 3.0) < 0.01
    print("metrics_utils: self-check ok")
