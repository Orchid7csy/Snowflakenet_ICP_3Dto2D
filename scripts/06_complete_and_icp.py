"""
scripts/06_complete_and_icp.py

流程（单样本 / 批量均可）：
1) 读取 processed_with_removal 的 input/obs/meta（以及可选 gt）
2) 用 SnowflakeNet 对 input 点云补全，保存到 data/completed/
3) 读取 meta 逆归一化：将补全点云从 input 坐标系逆变换回 obs_w 附近（粗配准）
4) 以粗配准结果为 source、obs_w 为 target，直接 ICP 精配准
5) 输出 ICP 位姿矩阵与配准误差指标（fitness / inlier_rmse），可选可视化

注意：
- 这里的“补全”是加载 checkpoint 做推理，并不调用 02_train_completion.py（它是训练脚本）。
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
import open3d as o3d
import torch

# ── 路径设置（与其他 scripts 保持一致）─────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_SNET_ROOT = os.path.join(_PROJECT_ROOT, "Snet", "SnowflakeNet-main")

for _p in (_PROJECT_ROOT, _SNET_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.model_completion import SnowflakeNet
from src.pose_estimation.icp import icp_refine


def _ensure_dir(path_or_file: str) -> str:
    """
    允许用户传入目录或误传成 .npy 文件路径：
    - 若以 .npy / .npz 结尾，则取其 dirname 作为输出目录。
    - 返回绝对路径目录。
    """
    p = os.path.abspath(path_or_file)
    if p.lower().endswith((".npy", ".npz")):
        p = os.path.dirname(p)
    return p


def load_snowflakenet(ckpt_path: str) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_loc = device if device.type == "cuda" else "cpu"

    model = SnowflakeNet(up_factors=[1, 4, 8])
    checkpoint = torch.load(ckpt_path, map_location=map_loc)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def complete_points(model: torch.nn.Module, input_points: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    pts = input_points.astype(np.float32)

    # SnowflakeNet 默认期望 2048
    if pts.shape[0] < 2048:
        idx = np.random.choice(pts.shape[0], 2048, replace=True)
        pts = pts[idx]
    elif pts.shape[0] > 2048:
        idx = np.random.choice(pts.shape[0], 2048, replace=False)
        pts = pts[idx]

    x = torch.from_numpy(pts).unsqueeze(0).to(device)
    outs = model(x)  # [Pc, P1, P2, P3]
    dense = outs[-1].squeeze(0).detach().cpu().numpy().astype(np.float32)
    return dense


def inverse_pca(points_aligned: np.ndarray, R_pca: np.ndarray, mu_pca: np.ndarray) -> np.ndarray:
    """
    前向（脚本里）: p_aligned = (R_pca @ (p - mu)^T)^T + mu
    逆向:           p        = (R_pca^T @ (p_aligned - mu)^T)^T + mu
    """
    R = np.asarray(R_pca, dtype=np.float64).reshape(3, 3)
    mu = np.asarray(mu_pca, dtype=np.float64).reshape(1, 3)
    X = np.asarray(points_aligned, dtype=np.float64) - mu
    return ((R.T @ X.T).T + mu).astype(np.float32)


def apply_inverse_normalization(p_comp: np.ndarray, meta: dict) -> np.ndarray:
    """
    将补全点云从 input 坐标系“瞬移”回 p_obs_w 附近（粗配准）。
    input 点云生成链路：
      p_norm = p_obs_w - C_bbox
      p_in   = PCA_align(p_norm) = (R_pca @ (p_norm - mu)) + mu
    逆链路：
      p_norm = invPCA(p_in)
      p_rough = p_norm + C_bbox
    """
    C_bbox = meta["C_bbox"].astype(np.float32).reshape(1, 3)
    R_pca = meta["R_pca"].astype(np.float32).reshape(3, 3)
    mu_pca = meta.get("mu_pca", np.zeros(3, dtype=np.float32)).astype(np.float32).reshape(3)

    p_norm = inverse_pca(p_comp, R_pca, mu_pca)
    p_rough = (p_norm + C_bbox).astype(np.float32)
    return p_rough


def to_pcd(points: np.ndarray, color: tuple[float, float, float] | None = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if color is not None:
        pcd.paint_uniform_color(list(color))
    return pcd


def run_one(
    stem: str,
    input_path: str,
    obs_path: str,
    meta_path: str,
    model: torch.nn.Module,
    completed_dir: str,
    icp_dist: float,
    icp_mode: str,
    icp_iter: int,
    vis: bool,
) -> dict:
    p_in = np.load(input_path).astype(np.float32)
    p_obs_w = np.load(obs_path).astype(np.float32)
    meta = dict(np.load(meta_path))

    p_comp = complete_points(model, p_in)
    completed_dir = _ensure_dir(completed_dir)
    os.makedirs(completed_dir, exist_ok=True)
    comp_path = os.path.join(completed_dir, f"{stem}_completed.npy")
    np.save(comp_path, p_comp.astype(np.float32))

    p_comp_rough = apply_inverse_normalization(p_comp, meta)

    src = to_pcd(p_comp_rough, color=(1.0, 0.6, 0.1))  # 橙：补全粗配准
    tgt = to_pcd(p_obs_w, color=(1.0, 0.0, 0.0))       # 红：真实观测 obs_w
    init = np.eye(4, dtype=np.float64)
    reg = icp_refine(
        source=src,
        target=tgt,
        init_transform=init,
        max_correspondence_distance=icp_dist,
        mode=icp_mode,
        max_iteration=icp_iter,
    )

    T = np.asarray(reg.transformation, dtype=np.float64)

    if vis:
        src_icp = src.transform(T.copy())
        src_icp.paint_uniform_color([0.25, 0.85, 0.35])  # 绿：ICP 后
        o3d.visualization.draw_geometries(
            [tgt, src_icp],
            window_name="ICP overlay (target=obs_w red, source=comp green after ICP)",
            width=1280,
            height=720,
        )

    result = {
        "stem": stem,
        "completed_path": comp_path,
        "icp_fitness": float(reg.fitness),
        "icp_inlier_rmse": float(reg.inlier_rmse),
        "T_icp": T,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Completion + inverse-normalization + ICP refine")
    parser.add_argument("--data-root", default=os.path.join(_PROJECT_ROOT, "data", "processed_with_removal"))
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--stem", default="airplane_0627_v2", help="不带扩展名，例如 airplane_0627_v2")
    parser.add_argument(
        "--top-view-idx",
        type=int,
        default=0,
        help="只补全/配准俯拍视角 v{idx}（与 01_preprocessing_with_removal.py 的固定俯拍约定一致，默认 v0）",
    )
    parser.add_argument("--ckpt", default=os.path.join(_PROJECT_ROOT, "checkpoints", "snet_finetune", "ckpt-airplane-best.pth"))
    parser.add_argument(
        "--completed-dir",
        default=os.path.join(_PROJECT_ROOT, "data", "completed"),
        help="补全点云输出目录（默认 <项目>/data/completed）；会保存为 <stem>_completed.npy",
    )
    parser.add_argument("--icp-dist", type=float, default=0.03, help="ICP max correspondence distance（归一化坐标系量级）")
    parser.add_argument("--icp-mode", default="point_to_plane", choices=("point_to_point", "point_to_plane"))
    parser.add_argument("--icp-iter", type=int, default=50)
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()

    # 允许从任意 cwd 运行：将相对路径统一解释为“相对项目根”
    if not os.path.isabs(args.data_root):
        args.data_root = os.path.abspath(os.path.join(_PROJECT_ROOT, args.data_root))
    if not os.path.isabs(args.ckpt):
        args.ckpt = os.path.abspath(os.path.join(_PROJECT_ROOT, args.ckpt))
    args.completed_dir = _ensure_dir(args.completed_dir)
    if not os.path.isabs(args.completed_dir):
        args.completed_dir = os.path.abspath(os.path.join(_PROJECT_ROOT, args.completed_dir))

    # 只处理俯拍 version：强制 stem 末尾为 _v{top_view_idx}
    suffix = f"_v{args.top_view_idx}"
    if not args.stem.endswith(suffix):
        args.stem = args.stem.split("_v")[0] + suffix

    split_root = os.path.join(args.data_root, args.split)
    input_path = os.path.join(split_root, "input", f"{args.stem}.npy")
    obs_path = os.path.join(split_root, "obs", f"{args.stem}.npy")
    meta_path = os.path.join(split_root, "meta", f"{args.stem}.npz")

    for p in (input_path, obs_path, meta_path, args.ckpt):
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p}\n"
                f"(data_root resolved to: {args.data_root})"
            )

    model = load_snowflakenet(args.ckpt)
    out = run_one(
        stem=args.stem,
        input_path=input_path,
        obs_path=obs_path,
        meta_path=meta_path,
        model=model,
        completed_dir=args.completed_dir,
        icp_dist=args.icp_dist,
        icp_mode=args.icp_mode,
        icp_iter=args.icp_iter,
        vis=args.vis,
    )

    T = out["T_icp"]
    print("=== ICP Result ===")
    print(f"stem: {out['stem']}")
    print(f"completed: {out['completed_path']}")
    print(f"fitness: {out['icp_fitness']:.6f}")
    print(f"inlier_rmse: {out['icp_inlier_rmse']:.6f}")
    print("T_icp (apply to comp_rough -> obs_w):")
    np.set_printoptions(precision=6, suppress=True)
    print(T)


if __name__ == "__main__":
    main()

