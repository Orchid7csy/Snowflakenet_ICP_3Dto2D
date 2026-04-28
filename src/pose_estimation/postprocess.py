"""
补全后处理：在「不读 GT / 不用 obs 监督」的前提下，抑制 Chamfer 类损失导致的稀疏飞点。

允许的信息（与训练时条件一致）:
  - 模型稠密预测 pred（P3 空间，与 input 同坐标系）；
  - 可选：残缺 input 点云，仅用于 (a) 估计采样尺度、 (b) 到 input 的最近距离门控的并集。

不使用的信息: 测试集 ground truth、obs 点云、以及任何在推理时本不应存在的真值。

默认策略: Open3D 统计离群点剔除 + 可选的「到 input 距离 < tau」与 pred 的并集
(tau = input_gate_mul * median(k-NN 距离于 input 上)）。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import open3d as o3d

# 固定默认超参（可在 val 上调，勿在 test 上自动搜参）
DEFAULT_SOR_NB = 20
DEFAULT_SOR_STD = 2.0
DEFAULT_INPUT_GATE_MUL = 4.0
DEFAULT_INPUT_KNN = 8
# 可选：要求点在半径邻域内足够稠密，进一步去掉孤立柱（与 SOR 与关系：最终 keep &= rad）
DEFAULT_RADIUS = 0.0
DEFAULT_RADIUS_NB = 16


@dataclass
class FilterConfig:
    sor_nb: int = DEFAULT_SOR_NB
    sor_std: float = DEFAULT_SOR_STD
    use_input_gate: bool = True
    input_knn: int = DEFAULT_INPUT_KNN
    input_gate_mul: float = DEFAULT_INPUT_GATE_MUL
    radius: float = DEFAULT_RADIUS
    radius_nb: int = DEFAULT_RADIUS_NB


def _median_kth_nn(input_xyz: np.ndarray, k: int) -> float:
    """在 input 上，每个点到第 k 近邻(不含自身)距离的全局中位数。input: (M,3)。"""
    pts = np.asarray(input_xyz, dtype=np.float64)
    n = pts.shape[0]
    if n < k + 1:
        return 0.02
    d2 = np.sum((pts[:, None, :] - pts[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d2, np.inf)
    d_sorted = np.sort(d2, axis=1)
    # 第 1 近邻: d_sorted[:,0], … 第 k 近邻: d_sorted[:, k-1]
    d_k = np.sqrt(d_sorted[:, k - 1].astype(np.float64))
    return float(np.median(d_k))


def _min_dist_to_reference(pred: np.ndarray, reference_xyz: np.ndarray) -> np.ndarray:
    """每个 pred 点到 reference 点集的最近欧氏距离，(N,)。"""
    p = np.asarray(pred, dtype=np.float64)
    x = np.asarray(reference_xyz, dtype=np.float64)
    try:
        from scipy.spatial import cKDTree

        t = cKDTree(x)
        try:
            d, _ = t.query(p, k=1, workers=-1)
        except TypeError:
            d, _ = t.query(p, k=1)
        return np.asarray(d, dtype=np.float64)
    except Exception:
        m = x.shape[0]
        out = np.empty(p.shape[0], dtype=np.float64)
        for s in range(0, p.shape[0], 4096):
            e = min(s + 4096, p.shape[0])
            b = p[s:e, None, :] - x[None, :, :]
            out[s:e] = np.sqrt(np.min(np.sum(b * b, axis=2), axis=1))
        return out


def filter_completion_spurious(
    pred: np.ndarray,
    input_points: Optional[np.ndarray] = None,
    *,
    cfg: Optional[FilterConfig] = None,
) -> Tuple[np.ndarray, dict[str, Any]]:
    """
    返回 (filtered_pred, info)。

    步骤:
      1) 对 pred 作 remove_statistical_outlier，得到行索引集合;
      2) 若 cfg.use_input_gate 且提供 input: tau = input_gate_mul * median_kth_nn(input),
         对全体 pred 标量最近距离 d(p,input)<tau 的标为真，与(1)并集;
      3) 若 cfg.radius>0: 对 pred 作 remove_radius_outlier，inlier 为稠密; 与 (1)(2) 的 keep 作交集。
    """
    cfg = cfg or FilterConfig()
    p = np.asarray(pred, dtype=np.float32)
    n0 = p.shape[0]
    if n0 == 0:
        return p, {
            "n_in": 0,
            "n_out": 0,
            "drop_ratio": 0.0,
            "sor_inliers": 0,
            "input_keep_extra": 0,
            "radius_inliers": n0,
        }

    pcd0 = o3d.geometry.PointCloud()
    pcd0.points = o3d.utility.Vector3dVector(p.astype(np.float64))

    pcd_s, ind_sor = pcd0.remove_statistical_outlier(
        cfg.sor_nb, std_ratio=cfg.sor_std
    )
    keep = np.zeros(n0, dtype=bool)
    keep[np.asarray(ind_sor, dtype=np.int64)] = True
    n_sor = int(keep.sum())

    tau: Optional[float] = None
    n_in_extra = 0
    if cfg.use_input_gate and input_points is not None and input_points.size > 0:
        scale = _median_kth_nn(
            input_points, k=min(cfg.input_knn, max(1, input_points.shape[0] - 1))
        )
        tau = max(scale * 1e-6, float(cfg.input_gate_mul) * scale)
        dmin = _min_dist_to_reference(p, input_points)
        near = dmin < tau
        n_in_extra = int((near & ~keep).sum())
        keep = keep | near

    n_rad = n0
    if cfg.radius and cfg.radius > 0.0:
        pcd1, ind_rad = pcd0.remove_radius_outlier(
            cfg.radius_nb, radius=cfg.radius
        )
        mrad = np.zeros(n0, dtype=bool)
        mrad[np.asarray(ind_rad, dtype=np.int64)] = True
        n_rad = int(mrad.sum())
        keep = keep & mrad

    n1 = int(keep.sum())
    info: dict[str, Any] = {
        "n_in": n0,
        "n_out": n1,
        "drop_ratio": float(1.0 - n1 / n0) if n0 else 0.0,
        "sor_inliers": n_sor,
        "input_keep_extra": n_in_extra,
        "input_gate_tau": tau,
        "radius_inliers": n_rad,
    }
    return p[keep], info


# ── Registration-aware 抗幻觉过滤 ─────────────────────────────────────────
# 与 filter_completion_spurious 的关键区别：
#   1) 作用在**与 obs 同坐标系**的补全结果上（世界系，逆归一化后）；
#   2) 门控语义为**硬切除**：d(p, obs) > tau 的点直接剔除，不与 SOR 作并集；
#   3) 其后再走一次**严格** SOR，只保留密度充足的核心骨架。
# 物理直觉：只信任网络在真实观测附近的「高置信延伸」；丢弃大面积悬空幻觉与散沙噪点。

DEFAULT_REG_GATE_MUL = 3.0
DEFAULT_REG_GATE_TAU_MODE = "comp_median"  # 或 "obs_knn"
DEFAULT_REG_GATE_OBS_KNN = 8
DEFAULT_REG_SOR_NB = 20
DEFAULT_REG_SOR_STD = 1.0


@dataclass
class RegistrationFilterConfig:
    gate_mul: float = DEFAULT_REG_GATE_MUL
    # "comp_median"：tau = gate_mul * median_i d(comp_i, obs)   —— 忠于需求原文
    # "obs_knn"    ：tau = gate_mul * median_kth_nn(obs, k)     —— 以 obs 采样尺度为锚
    gate_tau_mode: str = DEFAULT_REG_GATE_TAU_MODE
    gate_obs_knn: int = DEFAULT_REG_GATE_OBS_KNN
    sor_nb: int = DEFAULT_REG_SOR_NB
    sor_std: float = DEFAULT_REG_SOR_STD


def filter_registration_aware(
    comp_world: np.ndarray,
    obs_world: np.ndarray,
    *,
    cfg: Optional[RegistrationFilterConfig] = None,
) -> Tuple[np.ndarray, dict[str, Any]]:
    """
    Registration-aware 抗幻觉过滤（与 obs 同坐标系）。

    步骤:
      1) Input-Distance Gate: 对每个 comp_world 点计算到 obs_world 的最近欧氏距离 d_i；
         依据 cfg.gate_tau_mode 选择阈值 tau，硬切除 d_i > tau 的点。
      2) Strict SOR: 对门控保留的点云再做一次严格的 Open3D remove_statistical_outlier。

    返回 (filtered, info)。info 含 n_rough / n_after_gate / n_after_sor / tau / mode 等。
    """
    cfg = cfg or RegistrationFilterConfig()
    p = np.asarray(comp_world, dtype=np.float32)
    n0 = int(p.shape[0])

    info: dict[str, Any] = {
        "n_rough": n0,
        "n_after_gate": n0,
        "n_after_sor": n0,
        "n_out": n0,
        "gate_drop": 0,
        "sor_drop": 0,
        "drop_ratio": 0.0,
        "gate_tau": None,
        "gate_tau_mode": cfg.gate_tau_mode,
        "gate_mul": float(cfg.gate_mul),
    }
    if n0 == 0 or obs_world is None or np.asarray(obs_world).size == 0:
        return p, info

    # 1) 基于 obs 的距离门控
    dmin = _min_dist_to_reference(p, obs_world)
    if cfg.gate_tau_mode == "obs_knn":
        k = min(int(cfg.gate_obs_knn), max(1, int(obs_world.shape[0]) - 1))
        scale = _median_kth_nn(obs_world, k=k)
    else:
        scale = float(np.median(dmin)) if dmin.size > 0 else 0.0
    tau = float(cfg.gate_mul) * float(scale)
    tau = max(tau, 1e-12)  # 数值下界，避免全丢
    keep_gate = dmin <= tau
    n_after_gate = int(keep_gate.sum())
    info.update(
        {
            "gate_tau": tau,
            "gate_tau_scale": float(scale),
            "n_after_gate": n_after_gate,
            "gate_drop": n0 - n_after_gate,
        }
    )
    if n_after_gate == 0:
        info["n_out"] = 0
        info["drop_ratio"] = 1.0
        return p[:0], info
    p_gated = p[keep_gate]

    # 2) 严格 SOR
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p_gated.astype(np.float64))
    _, ind_sor = pcd.remove_statistical_outlier(
        int(cfg.sor_nb), std_ratio=float(cfg.sor_std)
    )
    ind_sor = np.asarray(ind_sor, dtype=np.int64)
    p_out = p_gated[ind_sor] if ind_sor.size > 0 else p_gated[:0]
    n_after_sor = int(p_out.shape[0])

    info.update(
        {
            "n_after_sor": n_after_sor,
            "sor_drop": n_after_gate - n_after_sor,
            "n_out": n_after_sor,
            "drop_ratio": float(1.0 - n_after_sor / n0) if n0 else 0.0,
        }
    )
    return p_out.astype(np.float32), info
