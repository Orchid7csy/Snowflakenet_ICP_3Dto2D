#!/usr/bin/env python3
"""
收敛域扩张管线可视化：连续两个 Open3D 窗格。

1) **Canonical**：``input``（残缺）vs ``P_pred_cano``（补全）——仅看几何补全质量。
2) **World**：``obs_w`` vs CAD@``T_final``（最终 ICP）——看位姿对齐。

反归一化与 ``scripts/05_estimate_pose.py`` 一致（行向量约定 ``p' = p @ R^T + t``）::

    P_pred_w = (P_pred_cano * scale_cano + centroid_cano) @ T_coarse[:3,:3]^T + T_coarse[:3,3]

等价写成齐次右乘： ``P_pred_w = homogeneous_row(P_pred_obj) @ T_coarse``（见 ``_apply_row_transform``）。

用法（与 ``05_estimate_pose.py --vis`` 参数对齐）::

  PYTHONPATH=. python scripts/07_viz_convergence_basin.py --stem <stem>
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))


def _load_pose_module():
    path = os.path.join(_SCRIPT_DIR, "05_estimate_pose.py")
    spec = importlib.util.spec_from_file_location("_pose_estimate_pose", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    pose = _load_pose_module()
    parser = argparse.ArgumentParser(
        description="Canonical + World 双窗格可视化（复用 05_estimate_pose.run_one）",
    )
    parser.add_argument(
        "--data-root",
        default=os.path.join(_PROJECT_ROOT, "data", "processed", "PCN_far8_cano_in2048_gt16384"),
    )
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument("--stem", required=True)
    parser.add_argument(
        "--ckpt",
        default=os.path.join(_PROJECT_ROOT, "checkpoints", "snet_finetune", "ckpt-best.pth"),
    )
    parser.add_argument("--completed-dir", default=os.path.join(_PROJECT_ROOT, "data", "completed"))
    parser.add_argument("--icp-dist", type=float, default=0.03)
    parser.add_argument("--icp-mode", default="point_to_plane", choices=("point_to_point", "point_to_plane"))
    parser.add_argument("--icp-iter", type=int, default=50)
    parser.add_argument("--fpfh-voxel", type=float, default=0.03)
    parser.add_argument("--gate-fitness", type=float, default=0.5)
    parser.add_argument("--input-resample", default="fps", choices=("fps", "random"))
    parser.add_argument("--no-reg-filter", action="store_true")
    parser.add_argument("--legacy-comp-filter", action="store_true")
    parser.add_argument("--gate-mul", type=float, default=3.0)
    parser.add_argument("--gate-tau-mode", default="comp_median", choices=("comp_median", "obs_knn"))
    parser.add_argument("--gate-obs-knn", type=int, default=8)
    args = parser.parse_args()

    if not os.path.isabs(args.data_root):
        args.data_root = os.path.abspath(os.path.join(_PROJECT_ROOT, args.data_root))
    if not os.path.isabs(args.ckpt):
        args.ckpt = os.path.abspath(os.path.join(_PROJECT_ROOT, args.ckpt))
    cd = args.completed_dir
    if not os.path.isabs(cd):
        cd = os.path.abspath(os.path.join(_PROJECT_ROOT, cd))
    pose._ensure_dir(cd)
    args.completed_dir = cd

    split_root = os.path.join(args.data_root, args.split)
    input_path = os.path.join(split_root, "input", f"{args.stem}.npy")
    obs_path = os.path.join(split_root, "obs_w", f"{args.stem}.npy")
    meta_path = os.path.join(split_root, "meta", f"{args.stem}.npz")
    for p in (input_path, obs_path, meta_path, args.ckpt):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

    rcfg = pose.RegistrationFilterConfig(
        gate_mul=args.gate_mul,
        gate_tau_mode=args.gate_tau_mode,
        gate_obs_knn=args.gate_obs_knn,
    )
    model = pose.load_snowflakenet(args.ckpt)
    pose.run_one(
        stem=args.stem,
        input_path=input_path,
        obs_path=obs_path,
        meta_path=meta_path,
        model=model,
        completed_dir=args.completed_dir,
        icp_dist=args.icp_dist,
        icp_mode=args.icp_mode,
        icp_iter=args.icp_iter,
        vis=True,
        fpfh_voxel=args.fpfh_voxel,
        gate_fitness=args.gate_fitness,
        input_resample_mode=args.input_resample,
        do_comp_filter=args.legacy_comp_filter,
        do_reg_filter=not args.no_reg_filter,
        reg_filter_cfg=rcfg,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
