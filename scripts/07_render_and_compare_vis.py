"""
scripts/07_render_and_compare_vis.py

在不修改 scripts/07_render_and_compare.py 的前提下，为 Render-and-Compare 结果添加 2D 可视化：
- 距离场（Distance Transform）热力图
- 叠加投影点：P_obs（作为 GT 边缘来源）、P_comp_before、P_comp_after

用法（pipeline）:
  python scripts/07_render_and_compare_vis.py --stem airplane_0627_v0

输出:
  data/render_compare_vis/<stem>_rc_vis.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# 动态加载 scripts/07_render_and_compare.py（不修改原文件，也不要求 scripts 成为包）
import importlib.util


def _load_rc_module():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rc_path = os.path.join(script_dir, "07_render_and_compare.py")
    spec = importlib.util.spec_from_file_location("render_and_compare", rc_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module from {rc_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[attr-defined]
    return module


rc = _load_rc_module()


def _project(P: np.ndarray, K: np.ndarray, T: np.ndarray) -> np.ndarray:
    """简单针孔投影：返回 (M,2)，过滤 z<=0。"""
    R = T[:3, :3].astype(np.float64)
    t = T[:3, 3].astype(np.float64)
    X = (R @ P.astype(np.float64).T + t.reshape(3, 1)).T
    mask = X[:, 2] > 0.01
    X = X[mask]
    uvw = (K.astype(np.float64) @ X.T).T
    uv = uvw[:, :2] / uvw[:, 2:3]
    return uv.astype(np.float32)


def _scatter(ax, uv: np.ndarray, H: int, W: int, color: str, label: str, s: float = 2.0, alpha: float = 0.8):
    if uv.size == 0:
        return
    m = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    uv = uv[m]
    ax.scatter(uv[:, 0], uv[:, 1], s=s, c=color, alpha=alpha, label=label, linewidths=0)


def run_pipeline_vis(args):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        raise RuntimeError("需要 matplotlib 用于 2D 可视化：pip install matplotlib") from e

    if not os.path.isabs(args.data_root):
        args.data_root = os.path.join(rc._PROJECT_ROOT, args.data_root)
    if not os.path.isabs(args.completed_dir):
        args.completed_dir = os.path.join(rc._PROJECT_ROOT, args.completed_dir)

    split_root = os.path.join(args.data_root, args.split)
    comp_path = os.path.join(args.completed_dir, f"{args.stem}_completed.npy")
    obs_path = os.path.join(split_root, "obs", f"{args.stem}.npy")
    meta_path = os.path.join(split_root, "meta", f"{args.stem}.npz")

    for p in (comp_path, obs_path, meta_path):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    P_comp = np.load(comp_path).astype(np.float32)
    P_obs_w = np.load(obs_path).astype(np.float32)
    meta = dict(np.load(meta_path))

    # inverse-normalization: comp → rough (near obs_w)
    P_comp_rough = rc.inverse_normalize(P_comp, meta)

    # 与 07 相同的“虚拟相机”：平移到 obs 质心附近并拉到 +2.5m
    centroid = P_obs_w.mean(axis=0)
    T_cam = np.eye(4, dtype=np.float64)
    T_cam[:3, 3] = -centroid.astype(np.float64)
    T_cam[2, 3] += 2.5

    P_obs_cam = (T_cam[:3, :3] @ P_obs_w.astype(np.float64).T + T_cam[:3, 3:4]).T.astype(np.float32)
    P_comp_cam = (T_cam[:3, :3] @ P_comp_rough.astype(np.float64).T + T_cam[:3, 3:4]).T.astype(np.float32)

    H, W = 480, 640
    K = rc.default_intrinsics()

    # dt 场来自 obs_cam 的投影边缘
    dt = rc.build_distance_field(P_obs_cam, K, np.eye(4, dtype=np.float64), H, W)
    dt_img = dt.squeeze(0).squeeze(0).cpu().numpy()

    # 跑一次优化，得到 T_final（camera frame）
    T_init = np.eye(4, dtype=np.float64)
    T_final, losses = rc.render_and_compare(
        P_comp_cam, dt, K, T_init, H, W,
        num_iters=args.num_iters, lr=args.lr, device=args.device,
    )

    # 2D 投影（obs 用 I，comp_before 用 I，comp_after 用 T_final）
    uv_obs = _project(P_obs_cam, K, np.eye(4, dtype=np.float64))
    uv_before = _project(P_comp_cam, K, np.eye(4, dtype=np.float64))
    uv_after = _project(P_comp_cam, K, T_final)

    out_dir = os.path.join(rc._PROJECT_ROOT, "data", "render_compare_vis")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.stem}_rc_vis.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=140)

    ax0 = axes[0]
    ax0.set_title("Distance Transform (from obs projection)")
    im = ax0.imshow(dt_img, cmap="magma", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)
    ax0.set_xlim([0, W - 1])
    ax0.set_ylim([H - 1, 0])
    ax0.axis("off")

    ax1 = axes[1]
    ax1.set_title("2D Overlay (same camera): obs / comp before / comp after")
    ax1.imshow(np.zeros((H, W), dtype=np.float32), cmap="gray", vmin=0, vmax=1)
    _scatter(ax1, uv_obs, H, W, color="#00ff66", label="obs edge pts", s=2.0, alpha=0.45)
    _scatter(ax1, uv_before, H, W, color="#ff3333", label="comp before", s=2.0, alpha=0.45)
    _scatter(ax1, uv_after, H, W, color="#33a1ff", label="comp after", s=2.0, alpha=0.75)
    ax1.set_xlim([0, W - 1])
    ax1.set_ylim([H - 1, 0])
    ax1.axis("off")
    ax1.legend(loc="lower left", framealpha=0.8, fontsize=9)

    fig.suptitle(f"{args.stem} | iters={args.num_iters} lr={args.lr}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    r_opt, t_opt = rc.pose_error(T_final, np.eye(4))
    print("=== R&C (pipeline) ===")
    print(f"saved: {out_path}")
    print(f"residual (camera frame): rot={r_opt:.4f}°  trans={t_opt:.6f}")
    np.set_printoptions(precision=6, suppress=True)
    print("T_final:")
    print(T_final)


def main():
    parser = argparse.ArgumentParser(description="Render-and-Compare pipeline + 2D visualization (backup script)")
    parser.add_argument("--data-root", default=os.path.join(rc._PROJECT_ROOT, "data", "processed_with_removal"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--stem", default="airplane_0627_v0")
    parser.add_argument("--completed-dir", default=os.path.join(rc._PROJECT_ROOT, "data", "completed"))
    parser.add_argument("--num-iters", type=int, default=200)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    run_pipeline_vis(args)


if __name__ == "__main__":
    main()

