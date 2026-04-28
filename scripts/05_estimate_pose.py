"""
补全 → 逆归一化（PCA + bbox）→ 可选 FPFH 粗配 → ICP；读 obs_w 与 meta（00_preprocess_pcn 产出）。

T_icp：将 filtered 补全点云对齐到 obs_w。
若需物体系位姿，可用 meta 中 T_far_4x4 与 T_icp 组合（见脚本输出说明）。

用法:
  python scripts/05_estimate_pose.py --data-root data/processed/PCN_far_cano_in2048_gt16384 --split test \\
    --stem pcn__test__plane_02691156__...__view_00 --ckpt checkpoints/snet_finetune/ckpt-best.pth
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict

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
from src.utils.io import to_o3d_pcd


def _ensure_dir(path_or_file: str) -> str:
    p = os.path.abspath(path_or_file)
    if p.lower().endswith((".npy", ".npz")):
        p = os.path.dirname(p)
    return p


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
    coarse: str,
    fpfh_voxel: float,
    *,
    do_comp_filter: bool = False,
    filter_cfg: FilterConfig | None = None,
    do_reg_filter: bool = True,
    reg_filter_cfg: RegistrationFilterConfig | None = None,
) -> dict:
    p_in = np.load(input_path).astype(np.float32)
    p_obs_w = np.load(obs_path).astype(np.float32)
    meta = dict(np.load(meta_path, allow_pickle=True))

    p_comp = complete_points(model, p_in)
    if do_comp_filter:
        cfg = filter_cfg or FilterConfig()
        p_comp, _ = filter_completion_spurious(p_comp, p_in, cfg=cfg)

    completed_dir = _ensure_dir(completed_dir)
    os.makedirs(completed_dir, exist_ok=True)
    comp_path = os.path.join(completed_dir, f"{stem}_completed.npy")
    np.save(comp_path, p_comp.astype(np.float32))

    p_comp_rough = prep.apply_inverse_normalization(p_comp, meta)

    filtered_comp = p_comp_rough
    if do_reg_filter:
        rcfg = reg_filter_cfg or RegistrationFilterConfig()
        filtered_comp, _ = filter_registration_aware(
            p_comp_rough, p_obs_w, cfg=rcfg
        )

    init = np.eye(4, dtype=np.float64)
    if coarse == "fpfh":
        vs = float(fpfh_voxel)
        sp = numpy_to_point_cloud(filtered_comp)
        tp = numpy_to_point_cloud(p_obs_w)
        sd, sf = voxel_downsample_with_fpfh(sp, vs)
        td, tf = voxel_downsample_with_fpfh(tp, vs)
        ransac = global_registration_fpfh_ransac(
            sd, td, sf, tf, vs, max_iterations=200_000, confidence=1000,
        )
        init = np.asarray(ransac.transformation, dtype=np.float64)
    src = to_o3d_pcd(filtered_comp, color=(1.0, 0.6, 0.1))
    tgt = to_o3d_pcd(p_obs_w, color=(1.0, 0.0, 0.0))
    reg = icp_refine(
        source=src, target=tgt, init_transform=init,
        max_correspondence_distance=icp_dist, mode=icp_mode, max_iteration=icp_iter,
    )
    t = np.asarray(reg.transformation, dtype=np.float64)
    if vis:
        src_icp = src.transform(t.copy())
        src_icp.paint_uniform_color([0.25, 0.85, 0.35])
        o3d.visualization.draw_geometries(
            [tgt, src_icp],
            window_name="ICP (target=obs_w red, source=comp green)",
            width=1280, height=720,
        )
    t_far = meta.get("T_far_4x4")
    return {
        "stem": stem, "completed_path": comp_path,
        "icp_fitness": float(reg.fitness), "icp_inlier_rmse": float(reg.inlier_rmse),
        "T_icp": t, "T_far_4x4": t_far, "meta_path": meta_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Completion + inv-norm + ICP pose")
    parser.add_argument(
        "--data-root",
        default=os.path.join(_PROJECT_ROOT, "data", "processed", "PCN_far_cano_in2048_gt16384"),
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
    parser.add_argument(
        "--coarse", choices=("none", "fpfh"), default="none",
        help="ICP 初值: none=单位阵; fpfh=先 FPFH+RANSAC 再 ICP",
    )
    parser.add_argument("--fpfh-voxel", type=float, default=0.03)
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
        coarse=args.coarse, fpfh_voxel=args.fpfh_voxel,
        do_comp_filter=args.legacy_comp_filter, do_reg_filter=not args.no_reg_filter,
        reg_filter_cfg=rcfg,
    )
    print("=== Result ===")
    print(f"stem: {out['stem']}")
    print(f"completed: {out['completed_path']}")
    print(f"fitness: {out['icp_fitness']:.6f}  inlier_rmse: {out['icp_inlier_rmse']:.6f}")
    print("T_icp (4x4, maps filtered completion -> obs_w):")
    np.set_printoptions(precision=6, suppress=True)
    print(out["T_icp"])
    if out.get("T_far_4x4") is not None:
        print("T_far_4x4 (row-vector rigid: p_w = p_obj @ R.T + t in homogeneous form):")
        print(out["T_far_4x4"])


if __name__ == "__main__":
    main()
