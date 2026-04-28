"""
3D 补全 + 位姿估计闭环：
  1) CAD(complete) ↔ obs_w 做 FPFH 粗配准，得 T_coarse；
  2) 用 T_coarse^-1 将 obs_w 摆正到粗物体系，再按 meta(centroid_cano/scale_cano)归一化到 canonical；
  3) canonical 输入固定 2048 点送入 SNet；
  4) 反归一化严格使用: P_pred_w = (P_pred_cano * S + C) * T_coarse；
  5) Gate-ICP: CAD ↔ P_pred_w，fitness 低于阈值时回退 T_coarse；
  6) 最终 ICP: CAD ↔ 原始 obs_w 锁定位姿。
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import open3d as o3d
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_SNET_ROOT = os.path.join(_PROJECT_ROOT, "Snet", "SnowflakeNet-main")
for _p in (_PROJECT_ROOT, _SNET_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data import preprocessing as prep
from src.models.snet_loader import complete_points, load_snowflakenet
from src.pose_estimation.fpfh import (
    global_registration_fpfh_ransac,
    numpy_to_point_cloud,
    voxel_downsample_with_fpfh,
)
from src.pose_estimation.icp import icp_refine
from src.pose_estimation.postprocess import (
    FilterConfig,
    RegistrationFilterConfig,
    filter_completion_spurious,
    filter_registration_aware,
)
from src.utils.io import read_pcd_xyz, to_o3d_pcd


def _ensure_dir(path_or_file: str) -> str:
    p = os.path.abspath(path_or_file)
    if p.lower().endswith((".npy", ".npz")):
        p = os.path.dirname(p)
    return p


def _apply_row_transform(points: np.ndarray, t: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    tt = np.asarray(t, dtype=np.float64).reshape(4, 4)
    return (p @ tt[:3, :3] + tt[:3, 3]).astype(np.float32)


def _invert_row_transform(t: np.ndarray) -> np.ndarray:
    tt = np.asarray(t, dtype=np.float64).reshape(4, 4)
    r = tt[:3, :3]
    trans = tt[:3, 3]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = r.T
    out[:3, 3] = -(trans @ r)
    return out


def _resample_2048(points: np.ndarray, seed: int = 0, mode: str = "fps") -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return prep.resample_fixed_n(
        np.asarray(points, dtype=np.float32), 2048, rng, mode=mode
    )


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
    fpfh_voxel: float,
    gate_fitness: float,
    *,
    input_resample_mode: str = "fps",
    do_comp_filter: bool = False,
    filter_cfg: FilterConfig | None = None,
    do_reg_filter: bool = True,
    reg_filter_cfg: RegistrationFilterConfig | None = None,
) -> dict:
    p_obs_w = np.load(obs_path).astype(np.float32)
    meta = dict(np.load(meta_path, allow_pickle=True))
    source_complete = str(meta.get("source_complete", ""))
    if not source_complete or not os.path.exists(source_complete):
        raise FileNotFoundError(f"meta.source_complete 不存在: {source_complete}")
    p_cad = read_pcd_xyz(source_complete).astype(np.float32)

    # 1) 粗配准（CAD -> obs_w）
    vs = float(fpfh_voxel)
    sp = numpy_to_point_cloud(p_cad)
    tp = numpy_to_point_cloud(p_obs_w)
    sd, sf = voxel_downsample_with_fpfh(sp, vs)
    td, tf = voxel_downsample_with_fpfh(tp, vs)
    ransac = global_registration_fpfh_ransac(
        sd, td, sf, tf, vs, max_iterations=200_000, confidence=1000,
    )
    t_coarse = np.asarray(ransac.transformation, dtype=np.float64)

    # 2) obs_w 摆正到粗物体系，再 canonical 归一化，并固定 2048 点
    t_coarse_inv = _invert_row_transform(t_coarse)
    p_rough = _apply_row_transform(p_obs_w, t_coarse_inv)
    centroid_cano = np.asarray(
        meta.get("centroid_cano", meta.get("C_cano")), dtype=np.float32
    ).reshape(1, 3)
    scale_cano = float(meta.get("scale_cano", 1.0))
    p_input_cano = ((p_rough - centroid_cano) / np.float32(scale_cano)).astype(np.float32)
    p_in = _resample_2048(p_input_cano, seed=0, mode=input_resample_mode)

    p_pred_cano = complete_points(model, p_in)
    if do_comp_filter:
        cfg = filter_cfg or FilterConfig()
        p_pred_cano, _ = filter_completion_spurious(p_pred_cano, p_in, cfg=cfg)

    completed_dir = _ensure_dir(completed_dir)
    os.makedirs(completed_dir, exist_ok=True)
    comp_path = os.path.join(completed_dir, f"{stem}_completed.npy")
    np.save(comp_path, p_pred_cano.astype(np.float32))

    # 3) 严格反归一化: P_pred_w = (P_pred_cano * S + C) * T_coarse
    p_pred_obj = (p_pred_cano * np.float32(scale_cano) + centroid_cano).astype(np.float32)
    p_pred_w = _apply_row_transform(p_pred_obj, t_coarse)

    filtered_comp = p_pred_w
    if do_reg_filter:
        rcfg = reg_filter_cfg or RegistrationFilterConfig()
        filtered_comp, _ = filter_registration_aware(
            p_pred_w, p_obs_w, cfg=rcfg
        )

    # 4) Gate-ICP（CAD -> P_pred_w），失败回退 T_coarse
    src_gate = to_o3d_pcd(p_cad, color=(0.2, 0.8, 1.0))
    tgt_gate = to_o3d_pcd(filtered_comp, color=(1.0, 0.6, 0.1))
    reg_gate = icp_refine(
        source=src_gate, target=tgt_gate, init_transform=t_coarse,
        max_correspondence_distance=icp_dist, mode=icp_mode, max_iteration=icp_iter,
    )
    gate_ok = float(reg_gate.fitness) >= float(gate_fitness)
    t_init = np.asarray(reg_gate.transformation if gate_ok else t_coarse, dtype=np.float64)

    # 5) 最终 ICP（CAD -> obs_w）
    src = to_o3d_pcd(p_cad, color=(0.2, 0.8, 1.0))
    tgt = to_o3d_pcd(p_obs_w, color=(1.0, 0.0, 0.0))
    reg = icp_refine(
        source=src, target=tgt, init_transform=t_init,
        max_correspondence_distance=icp_dist, mode=icp_mode, max_iteration=icp_iter,
    )
    t = np.asarray(reg.transformation, dtype=np.float64)
    if vis:
        src_icp = to_o3d_pcd(p_cad, color=(0.25, 0.85, 0.35))
        src_icp.transform(t.copy())
        o3d.visualization.draw_geometries(
            [tgt, src_icp],
            window_name="World space: obs_w(red) vs CAD@T_final(green)",
            width=1280, height=720,
        )
        cano_obs = to_o3d_pcd(p_in, color=(1.0, 0.0, 0.0))
        cano_pred = to_o3d_pcd(p_pred_cano, color=(0.25, 0.85, 0.35))
        o3d.visualization.draw_geometries(
            [cano_obs, cano_pred],
            window_name="Canonical space: input(red) vs completion(green)",
            width=1280, height=720,
        )
    t_far = meta.get("T_far_4x4")
    return {
        "stem": stem, "completed_path": comp_path,
        "icp_fitness": float(reg.fitness), "icp_inlier_rmse": float(reg.inlier_rmse),
        "T_coarse": t_coarse,
        "T_gate": np.asarray(reg_gate.transformation, dtype=np.float64),
        "gate_fitness": float(reg_gate.fitness),
        "gate_threshold": float(gate_fitness),
        "gate_accepted": bool(gate_ok),
        "T_icp": t, "T_far_4x4": t_far, "meta_path": meta_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Completion + inv-norm + ICP pose")
    parser.add_argument(
        "--data-root",
        default=os.path.join(_PROJECT_ROOT, "data", "processed", "PCN_far8_cano_in2048_gt16384"),
    )
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--stem", required=True, help="样本主名，无 .npy")
    parser.add_argument(
        "--ckpt",
        default=os.path.join(_PROJECT_ROOT, "checkpoints", "snet_finetune", "ckpt-best.pth"),
    )
    parser.add_argument("--completed-dir", default=os.path.join(_PROJECT_ROOT, "data", "completed"))
    parser.add_argument("--icp-dist", type=float, default=0.03)
    parser.add_argument("--icp-mode", default="point_to_plane", choices=("point_to_point", "point_to_plane"))
    parser.add_argument("--icp-iter", type=int, default=50)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--fpfh-voxel", type=float, default=0.03)
    parser.add_argument("--gate-fitness", type=float, default=0.5)
    parser.add_argument(
        "--input-resample",
        default="fps",
        choices=("fps", "random"),
        help="送入 SNet 前将 canonical 观测重采样到 2048 点的策略（与预处理一致可复现对齐）",
    )
    parser.add_argument("--no-reg-filter", action="store_true")
    parser.add_argument("--legacy-comp-filter", action="store_true", help="补全后归一化域 SOR+gate")
    parser.add_argument("--gate-mul", type=float, default=3.0)
    parser.add_argument("--gate-tau-mode", default="comp_median", choices=("comp_median", "obs_knn"))
    parser.add_argument("--gate-obs-knn", type=int, default=8)
    args = parser.parse_args()

    if not os.path.isabs(args.data_root):
        args.data_root = os.path.abspath(os.path.join(_PROJECT_ROOT, args.data_root))
    if not os.path.isabs(args.ckpt):
        args.ckpt = os.path.abspath(os.path.join(_PROJECT_ROOT, args.ckpt))
    args.completed_dir = _ensure_dir(args.completed_dir)
    if not os.path.isabs(args.completed_dir):
        args.completed_dir = os.path.abspath(os.path.join(_PROJECT_ROOT, args.completed_dir))

    split_root = os.path.join(args.data_root, args.split)
    input_path = os.path.join(split_root, "input", f"{args.stem}.npy")
    obs_path = os.path.join(split_root, "obs_w", f"{args.stem}.npy")
    meta_path = os.path.join(split_root, "meta", f"{args.stem}.npz")
    for p in (input_path, obs_path, meta_path, args.ckpt):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    rcfg = RegistrationFilterConfig(
        gate_mul=args.gate_mul, gate_tau_mode=args.gate_tau_mode, gate_obs_knn=args.gate_obs_knn,
    )
    model = load_snowflakenet(args.ckpt)
    out = run_one(
        stem=args.stem, input_path=input_path, obs_path=obs_path, meta_path=meta_path,
        model=model, completed_dir=args.completed_dir, icp_dist=args.icp_dist,
        icp_mode=args.icp_mode, icp_iter=args.icp_iter, vis=args.vis,
        fpfh_voxel=args.fpfh_voxel, gate_fitness=args.gate_fitness,
        input_resample_mode=args.input_resample,
        do_comp_filter=args.legacy_comp_filter, do_reg_filter=not args.no_reg_filter,
        reg_filter_cfg=rcfg,
    )
    print("=== Result ===")
    print(f"stem: {out['stem']}")
    print(f"completed: {out['completed_path']}")
    print(f"fitness: {out['icp_fitness']:.6f}  inlier_rmse: {out['icp_inlier_rmse']:.6f}")
    print("T_coarse (4x4, CAD -> obs_w, FPFH-RANSAC):")
    print(out["T_coarse"])
    print(
        f"gate: fitness={out['gate_fitness']:.6f} "
        f"threshold={out['gate_threshold']:.3f} accepted={out['gate_accepted']}"
    )
    print("T_icp (4x4, CAD -> obs_w, final ICP):")
    np.set_printoptions(precision=6, suppress=True)
    print(out["T_icp"])
    if out.get("T_far_4x4") is not None:
        print("T_far_4x4 (row-vector rigid: p_w = p_obj @ R.T + t in homogeneous form):")
        print(out["T_far_4x4"])


if __name__ == "__main__":
    main()
