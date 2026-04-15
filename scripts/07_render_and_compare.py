"""
scripts/07_render_and_compare.py

纯 PyTorch 可微 Render-and-Compare 位姿精对齐。
不依赖 PyTorch3D / 任何渲染引擎，仅用 PyTorch 张量操作 + scipy EDT。

两种运行模式:

  synthetic （默认）
    用同一 P_gt 构造已知 T_gt，加噪得到 T_init，优化并检验收敛精度。
    python scripts/07_render_and_compare.py

  pipeline
    读取 completed/ 与 obs/，以 T_icp (或 identity) 为初值做 R&C 精细化。
    python scripts/07_render_and_compare.py --mode pipeline --stem airplane_0627_v0
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_SNET_ROOT = os.path.join(_PROJECT_ROOT, "Snet", "SnowflakeNet-main")

for _p in (_PROJECT_ROOT, _SNET_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ═══════════════════════════════════════════════════════════════════════════
# 1. Rodrigues axis-angle ↔ rotation matrix  (全部可微)
# ═══════════════════════════════════════════════════════════════════════════

def _skew(v: torch.Tensor) -> torch.Tensor:
    z = torch.zeros_like(v[0])
    return torch.stack([
        torch.stack([z, -v[2], v[1]]),
        torch.stack([v[2], z, -v[0]]),
        torch.stack([-v[1], v[0], z]),
    ])


def rodrigues(omega: torch.Tensor) -> torch.Tensor:
    """axis-angle (3,) → SO(3) (3,3)，可微，theta→0 时一阶近似。"""
    theta = torch.norm(omega)
    if theta.item() < 1e-8:
        return torch.eye(3, device=omega.device, dtype=omega.dtype) + _skew(omega)
    K = _skew(omega / theta)
    return (
        torch.eye(3, device=omega.device, dtype=omega.dtype)
        + torch.sin(theta) * K
        + (1.0 - torch.cos(theta)) * (K @ K)
    )


def rodrigues_np(omega: np.ndarray) -> np.ndarray:
    """NumPy 版本（非可微，用于数据准备）。"""
    omega = np.asarray(omega, dtype=np.float64)
    theta = float(np.linalg.norm(omega))
    if theta < 1e-8:
        return np.eye(3, dtype=np.float64)
    k = omega / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return (np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)).astype(np.float64)


def rotmat_to_axis_angle(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    theta = float(np.arccos(trace))
    if theta < 1e-6:
        return np.zeros(3, dtype=np.float32)
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    w = w / (2.0 * np.sin(theta) + 1e-12) * theta
    return w.astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 2. 相机投影与距离场
# ═══════════════════════════════════════════════════════════════════════════

def default_intrinsics(
    fx: float = 500.0, fy: float = 500.0, cx: float = 320.0, cy: float = 240.0,
) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)


def project_np(P: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray):
    """NumPy 投影，返回 (M,2) 像素坐标和有效掩码。"""
    pts_cam = (R @ P.T + t.reshape(3, 1)).T
    valid = pts_cam[:, 2] > 0.01
    pts_v = pts_cam[valid]
    proj = (K @ pts_v.T).T
    uv = proj[:, :2] / proj[:, 2:3]
    return uv.astype(np.float32), valid


def build_distance_field(
    P_ref: np.ndarray,
    K: np.ndarray,
    T_gt: np.ndarray,
    H: int,
    W: int,
    max_dist: float = 50.0,
) -> torch.Tensor:
    """
    将 P_ref 按 T_gt 投影到 (H,W) 像素平面 → 二值边缘图 → EDT → 截断归一化。
    返回 (1,1,H,W) float32 张量。
    """
    R = T_gt[:3, :3].astype(np.float64)
    t = T_gt[:3, 3].astype(np.float64)
    uv, _ = project_np(P_ref.astype(np.float64), K.astype(np.float64), R, t)

    edge = np.zeros((H, W), dtype=np.uint8)
    u_px = np.clip(np.round(uv[:, 0]).astype(int), 0, W - 1)
    v_px = np.clip(np.round(uv[:, 1]).astype(int), 0, H - 1)
    edge[v_px, u_px] = 1

    dt = distance_transform_edt(1 - edge).astype(np.float32)
    dt = np.minimum(dt, max_dist) / max_dist
    return torch.from_numpy(dt).unsqueeze(0).unsqueeze(0)


def sample_dt(
    dt_tensor: torch.Tensor, uv: torch.Tensor, H: int, W: int,
) -> torch.Tensor:
    """
    可微双线性采样：在距离场 (1,1,H,W) 的 (u,v) 处取值，返回 (N,)。
    """
    u_norm = 2.0 * uv[:, 0] / (W - 1) - 1.0
    v_norm = 2.0 * uv[:, 1] / (H - 1) - 1.0
    grid = torch.stack([u_norm, v_norm], dim=-1).unsqueeze(0).unsqueeze(0)
    out = F.grid_sample(
        dt_tensor, grid, mode="bilinear", padding_mode="border", align_corners=True,
    )
    return out.reshape(-1)


# ═══════════════════════════════════════════════════════════════════════════
# 3. 可微优化主循环
# ═══════════════════════════════════════════════════════════════════════════

def render_and_compare(
    P_model: np.ndarray,
    dt_field: torch.Tensor,
    K_np: np.ndarray,
    T_init: np.ndarray,
    H: int = 480,
    W: int = 640,
    num_iters: int = 200,
    lr: float = 5e-4,
    device: str = "cpu",
) -> tuple[np.ndarray, list[float]]:
    """
    返回 (T_optimized (4,4), loss_history)。
    """
    dev = torch.device(device)
    P = torch.from_numpy(P_model.astype(np.float32)).to(dev)
    K = torch.from_numpy(K_np.astype(np.float32)).to(dev)
    dt = dt_field.to(dev)

    omega = torch.tensor(
        rotmat_to_axis_angle(T_init[:3, :3]),
        dtype=torch.float32, device=dev, requires_grad=True,
    )
    t = torch.tensor(
        T_init[:3, 3].astype(np.float32),
        dtype=torch.float32, device=dev, requires_grad=True,
    )

    optimizer = torch.optim.Adam([omega, t], lr=lr)
    losses: list[float] = []

    for i in range(num_iters):
        optimizer.zero_grad()
        R = rodrigues(omega)

        pts_cam = (R @ P.T + t.unsqueeze(1)).T          # (N, 3)
        z = pts_cam[:, 2]
        w = torch.sigmoid(200.0 * (z - 0.01))           # soft z>0 mask

        proj = (K @ pts_cam.T).T                         # (N, 3) homogeneous
        z_safe = proj[:, 2:3].clamp(min=0.01)
        uv = proj[:, :2] / z_safe                        # (N, 2) pixel

        d = sample_dt(dt, uv, H, W)
        loss = (d * w).sum() / (w.sum() + 1e-8)

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  iter {i + 1:4d}/{num_iters} | loss {loss.item():.6f}")

    with torch.no_grad():
        R_f = rodrigues(omega).cpu().numpy()
        t_f = t.cpu().numpy()

    T_final = np.eye(4, dtype=np.float64)
    T_final[:3, :3] = R_f
    T_final[:3, 3] = t_f
    return T_final, losses


# ═══════════════════════════════════════════════════════════════════════════
# 4. 误差指标
# ═══════════════════════════════════════════════════════════════════════════

def pose_error(T_est: np.ndarray, T_gt: np.ndarray) -> tuple[float, float]:
    """返回 (旋转误差°, 平移误差)。"""
    dR = T_est[:3, :3] @ T_gt[:3, :3].T
    trace = np.clip((np.trace(dR) - 1.0) / 2.0, -1.0, 1.0)
    rot_err = float(np.degrees(np.arccos(trace)))
    trans_err = float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))
    return rot_err, trans_err


# ═══════════════════════════════════════════════════════════════════════════
# 5. 管线辅助：逆归一化（与 06 脚本一致）
# ═══════════════════════════════════════════════════════════════════════════

def _inverse_pca(pts: np.ndarray, R_pca: np.ndarray, mu: np.ndarray) -> np.ndarray:
    R = np.asarray(R_pca, dtype=np.float64).reshape(3, 3)
    m = np.asarray(mu, dtype=np.float64).reshape(1, 3)
    return ((R.T @ (pts.astype(np.float64) - m).T).T + m).astype(np.float32)


def inverse_normalize(p_comp: np.ndarray, meta: dict) -> np.ndarray:
    C = meta["C_bbox"].astype(np.float32).reshape(1, 3)
    R_pca = meta["R_pca"].astype(np.float32).reshape(3, 3)
    mu = meta.get("mu_pca", np.zeros(3, dtype=np.float32)).astype(np.float32)
    return (_inverse_pca(p_comp, R_pca, mu) + C).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# 6. Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Render-and-Compare 位姿精对齐验证（纯 PyTorch）",
    )
    parser.add_argument("--mode", choices=("synthetic", "pipeline"), default="synthetic")
    parser.add_argument("--data-root", default=os.path.join(_PROJECT_ROOT, "data", "processed_with_removal"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--stem", default="airplane_0627_v0")
    parser.add_argument("--completed-dir", default=os.path.join(_PROJECT_ROOT, "data", "completed"))
    parser.add_argument("--num-iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--perturb-rot-deg", type=float, default=5.0)
    parser.add_argument("--perturb-trans", type=float, default=0.03)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if not os.path.isabs(args.data_root):
        args.data_root = os.path.join(_PROJECT_ROOT, args.data_root)
    if not os.path.isabs(args.completed_dir):
        args.completed_dir = os.path.join(_PROJECT_ROOT, args.completed_dir)

    H, W = 480, 640
    K = default_intrinsics()

    if args.mode == "synthetic":
        _run_synthetic(args, K, H, W)
    else:
        _run_pipeline(args, K, H, W)


def _run_synthetic(args, K, H, W):
    print("=" * 60)
    print("  Render-and-Compare: Synthetic Verification")
    print("=" * 60)

    gt_path = os.path.join(args.data_root, args.split, "gt", f"{args.stem}.npy")
    if os.path.exists(gt_path):
        P = np.load(gt_path).astype(np.float32)
        print(f"Loaded: {gt_path} ({P.shape[0]} pts)")
    else:
        print("GT not found, generating random unit-sphere cloud (2048 pts)")
        rng = np.random.default_rng(42)
        P = rng.standard_normal((2048, 3)).astype(np.float32)
        P /= np.linalg.norm(P, axis=1, keepdims=True)
        P *= 0.5

    T_gt = np.eye(4, dtype=np.float64)
    T_gt[2, 3] = 2.5

    axis = np.random.randn(3).astype(np.float64)
    axis /= np.linalg.norm(axis) + 1e-8
    angle = np.deg2rad(args.perturb_rot_deg)
    R_pert = rodrigues_np(axis * angle)
    t_pert = np.random.randn(3).astype(np.float64) * args.perturb_trans

    T_init = np.eye(4, dtype=np.float64)
    T_init[:3, :3] = R_pert @ T_gt[:3, :3]
    T_init[:3, 3] = T_gt[:3, 3] + t_pert

    r0, t0 = pose_error(T_init, T_gt)
    print(f"T_gt   : R=I, t=[0, 0, 2.5]")
    print(f"T_init : perturb rot={args.perturb_rot_deg}° trans={args.perturb_trans}")
    print(f"Initial error  : rot={r0:.4f}°  trans={t0:.6f}")
    print("-" * 60)

    dt = build_distance_field(P, K, T_gt, H, W)
    print(f"Distance field built: ({H}x{W}), edge pixels from {P.shape[0]} projected pts")
    print("-" * 60)

    T_final, losses = render_and_compare(
        P, dt, K, T_init, H, W,
        num_iters=args.num_iters, lr=args.lr, device=args.device,
    )

    r1, t1 = pose_error(T_final, T_gt)
    print()
    print("=" * 60)
    print(f"  Before R&C : rot_err = {r0:.4f}°   trans_err = {t0:.6f}")
    print(f"  After  R&C : rot_err = {r1:.4f}°   trans_err = {t1:.6f}")
    print(f"  Improvement: rot {r0 - r1:+.4f}°   trans {t0 - t1:+.6f}")
    print("=" * 60)
    np.set_printoptions(precision=6, suppress=True)
    print("T_gt:")
    print(T_gt)
    print("T_final:")
    print(T_final)


def _run_pipeline(args, K, H, W):
    print("=" * 60)
    print("  Render-and-Compare: Pipeline Mode")
    print("=" * 60)

    split_root = os.path.join(args.data_root, args.split)
    comp_path = os.path.join(args.completed_dir, f"{args.stem}_completed.npy")
    obs_path = os.path.join(split_root, "obs", f"{args.stem}.npy")
    meta_path = os.path.join(split_root, "meta", f"{args.stem}.npz")
    gt_path = os.path.join(split_root, "gt", f"{args.stem}.npy")

    for p in (comp_path, obs_path, meta_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    P_comp = np.load(comp_path).astype(np.float32)
    P_obs_w = np.load(obs_path).astype(np.float32)
    meta = dict(np.load(meta_path))

    P_comp_rough = inverse_normalize(P_comp, meta)
    print(f"P_comp: {P_comp.shape[0]} pts  →  P_comp_rough (inverse-normalized)")
    print(f"P_obs_w: {P_obs_w.shape[0]} pts")

    centroid = P_obs_w.mean(axis=0)
    T_cam = np.eye(4, dtype=np.float64)
    T_cam[:3, 3] = -centroid.astype(np.float64)
    T_cam[2, 3] += 2.5

    P_obs_cam = (T_cam[:3, :3] @ P_obs_w.astype(np.float64).T + T_cam[:3, 3:4]).T
    P_comp_cam = (T_cam[:3, :3] @ P_comp_rough.astype(np.float64).T + T_cam[:3, 3:4]).T

    dt = build_distance_field(
        P_obs_cam.astype(np.float32), K, np.eye(4, dtype=np.float64), H, W,
    )

    T_init = np.eye(4, dtype=np.float64)
    print(f"Camera: shifted to centroid + 2.5 along Z")
    print("-" * 60)

    T_final, losses = render_and_compare(
        P_comp_cam.astype(np.float32), dt, K, T_init, H, W,
        num_iters=args.num_iters, lr=args.lr, device=args.device,
    )

    r_opt, t_opt = pose_error(T_final, np.eye(4))
    print()
    print("=" * 60)
    print(f"  R&C residual: rot={r_opt:.4f}°  trans={t_opt:.6f}")

    if os.path.exists(gt_path):
        P_gt = np.load(gt_path).astype(np.float32)
        P_gt_cam = (T_cam[:3, :3] @ P_gt.astype(np.float64).T + T_cam[:3, 3:4]).T

        T_gt_relative = np.eye(4, dtype=np.float64)
        r_before, t_before = pose_error(T_init, T_gt_relative)
        r_after, t_after = pose_error(T_final, T_gt_relative)
        print(f"  vs GT: before rot={r_before:.4f}° trans={t_before:.6f}")
        print(f"  vs GT: after  rot={r_after:.4f}° trans={t_after:.6f}")
    print("=" * 60)

    np.set_printoptions(precision=6, suppress=True)
    print("T_final (camera-frame refinement):")
    print(T_final)


if __name__ == "__main__":
    main()
