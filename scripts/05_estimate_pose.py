"""
3D 补全 + 位姿估计闭环：
  1) CAD(complete) ↔ obs_w 做 FPFH 粗配准，得 T_coarse；
  2) 用 T_coarse^-1 将 obs_w 摆正到粗物体系，再按 meta(centroid_cano/scale_cano)归一化到 canonical；
  3) canonical 输入固定 2048 点送入 SNet；
  4) 反归一化严格使用: P_pred_w = (P_pred_cano * S + C) * T_coarse；
  5) Gate-ICP: CAD ↔ P_pred_w，fitness 低于阈值时回退 T_coarse；
  6) 最终 ICP: CAD ↔ 原始 obs_w 锁定位姿。

列举可选 stem::

  python scripts/05_estimate_pose.py --data-root data/processed/PCN_far8_cano_in2048_gt16384 \\
      --split test --list-stems

整 split 评估并按 PCN 类汇总::

  python scripts/05_estimate_pose.py --eval-all --split test \\
      --data-root data/processed/PCN_far8_cano_in2048_gt16384 \\
      --ckpt checkpoints/snet_finetune/ckpt-best.pth
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from typing import DefaultDict, Dict, List

import numpy as np
import open3d as o3d
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_SNET_ROOT = os.path.join(_PROJECT_ROOT, "Snet", "SnowflakeNet-main")
for _p in (_PROJECT_ROOT, _SNET_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.evaluation.cd_l1 import TAXONOMY_TABLE1, english_name_from_stem  # noqa: E402
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


def discover_valid_stems(data_root: str, split: str) -> List[str]:
    """``input/``、``obs_w/``、``meta/`` 均存在同名文件的 stem 列表（排序）。"""
    split_root = os.path.join(data_root, split)
    inp_dir = os.path.join(split_root, "input")
    if not os.path.isdir(inp_dir):
        return []
    stems: List[str] = []
    for fn in sorted(os.listdir(inp_dir)):
        if not fn.endswith(".npy"):
            continue
        stem = fn[:-4]
        ip = os.path.join(inp_dir, fn)
        ow = os.path.join(split_root, "obs_w", f"{stem}.npy")
        mz = os.path.join(split_root, "meta", f"{stem}.npz")
        if os.path.isfile(ip) and os.path.isfile(ow) and os.path.isfile(mz):
            stems.append(stem)
    return stems


def print_batch_summary(rows: List[Dict[str, object]]) -> None:
    """按 PCN 类汇总 icp_fitness / inlier_rmse / gate 接受率；Macro=各类均值，Micro=全体样本均值。"""
    if not rows:
        print("无成功样本。")
        return
    by_class: DefaultDict[str, List[Dict[str, object]]] = defaultdict(list)
    for r in rows:
        by_class[str(r["class"])].append(r)

    print("\n" + "=" * 92)
    print(
        f"{'Class':<12} {'n':>6} {'icp_fitness':>14} {'inlier_rmse':>14} {'gate_ok%':>10}  "
        f"(各类均为该类样本均值)"
    )
    print("-" * 92)

    class_mean_fit: List[float] = []
    class_mean_rmse: List[float] = []

    for _syn, cname in TAXONOMY_TABLE1:
        rs = by_class.get(cname, [])
        if not rs:
            print(f"{cname:<12} {0:6d} {'--':>14} {'--':>14} {'--':>10}")
            continue
        n = len(rs)
        mf = float(np.mean([float(x["icp_fitness"]) for x in rs]))
        mr = float(np.mean([float(x["icp_inlier_rmse"]) for x in rs]))
        gk = 100.0 * float(
            np.mean([1.0 if bool(x["gate_accepted"]) else 0.0 for x in rs])
        )
        class_mean_fit.append(mf)
        class_mean_rmse.append(mr)
        print(f"{cname:<12} {n:6d} {mf:14.6f} {mr:14.6f} {gk:9.1f}%")

    unk = [k for k in by_class if k not in {t[1] for t in TAXONOMY_TABLE1}]
    for u in sorted(unk):
        rs = by_class[u]
        n = len(rs)
        mf = float(np.mean([float(x["icp_fitness"]) for x in rs]))
        mr = float(np.mean([float(x["icp_inlier_rmse"]) for x in rs]))
        gk = 100.0 * float(
            np.mean([1.0 if bool(x["gate_accepted"]) else 0.0 for x in rs])
        )
        print(f"{u:<12} {n:6d} {mf:14.6f} {mr:14.6f} {gk:9.1f}%")

    print("-" * 92)
    macro_f = float(np.mean(class_mean_fit)) if class_mean_fit else float("nan")
    macro_r = float(np.mean(class_mean_rmse)) if class_mean_rmse else float("nan")
    micro_f = float(np.mean([float(r["icp_fitness"]) for r in rows]))
    micro_r = float(np.mean([float(r["icp_inlier_rmse"]) for r in rows]))
    micro_g = 100.0 * float(
        np.mean([1.0 if bool(r["gate_accepted"]) else 0.0 for r in rows])
    )
    n_all = len(rows)

    print(
        f"{'MacroAvg':<12} {'—':>6} {macro_f:14.6f} {macro_r:14.6f} {'—':>10}  "
        f"#classes={len(class_mean_fit)}"
    )
    print(
        f"{'MicroAvg':<12} {n_all:6d} {micro_f:14.6f} {micro_r:14.6f} {micro_g:9.1f}%  "
        "(全体样本；含 unknown 等非表 8 类)"
    )
    print("=" * 92 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Completion + inv-norm + ICP pose；单样本或 --eval-all 全班统计。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        default=os.path.join(_PROJECT_ROOT, "data", "processed", "PCN_far8_cano_in2048_gt16384"),
    )
    parser.add_argument("--split", default="test", choices=("train", "val", "test"))
    parser.add_argument(
        "--stem",
        default=None,
        help="单条样本主名（无 .npy）；与 --eval-all 互斥（eval-all 时忽略）",
    )
    parser.add_argument(
        "--eval-all",
        action="store_true",
        help="处理 split 下全部可用 stem（需在 input / obs_w / meta 下均有同名文件）",
    )
    parser.add_argument(
        "--list-stems",
        action="store_true",
        help="仅列出可用 stem（每行一个）并退出，不加载模型",
    )
    parser.add_argument(
        "--max-stems",
        type=int,
        default=0,
        help="--eval-all 时最多处理的样本数（0 表示不截断）",
    )
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

    stems_all = discover_valid_stems(args.data_root, args.split)

    if args.list_stems:
        print(f"# {args.split}: {len(stems_all)} stems (have input + obs_w + meta)")
        try:
            for st in stems_all:
                print(st)
        except BrokenPipeError:
            # ``... | head`` 会在读完行数后关闭管道，随后 write 触发该异常（退出码 0 即可）
            pass
        return

    rcfg = RegistrationFilterConfig(
        gate_mul=args.gate_mul,
        gate_tau_mode=args.gate_tau_mode,
        gate_obs_knn=args.gate_obs_knn,
    )

    try:
        from tqdm import tqdm
    except ImportError:

        def tqdm(x, **_k):
            return x

    if args.eval_all:
        if args.stem:
            print("[WARN] 已指定 --eval-all，忽略 --stem", file=sys.stderr)
        if args.vis:
            print("[WARN] --eval-all 时忽略 --vis（避免弹数千次窗）", file=sys.stderr)

        stems = list(stems_all)
        if args.max_stems and args.max_stems > 0:
            stems = stems[: args.max_stems]

        if not stems:
            print(f"未发现可用样本：检查 {args.data_root}/{args.split}/input|obs_w|meta", file=sys.stderr)
            sys.exit(1)

        if not os.path.exists(args.ckpt):
            raise FileNotFoundError(args.ckpt)

        print(f"eval-all: split={args.split}  stems={len(stems)}  ckpt={args.ckpt}")
        model = load_snowflakenet(args.ckpt)

        rows: List[Dict[str, object]] = []
        n_skip = 0
        split_root = os.path.join(args.data_root, args.split)

        for stem in tqdm(stems, desc="pose_icp"):
            inp = os.path.join(split_root, "input", f"{stem}.npy")
            ow = os.path.join(split_root, "obs_w", f"{stem}.npy")
            mz = os.path.join(split_root, "meta", f"{stem}.npz")
            try:
                out = run_one(
                    stem=stem,
                    input_path=inp,
                    obs_path=ow,
                    meta_path=mz,
                    model=model,
                    completed_dir=args.completed_dir,
                    icp_dist=args.icp_dist,
                    icp_mode=args.icp_mode,
                    icp_iter=args.icp_iter,
                    vis=False,
                    fpfh_voxel=args.fpfh_voxel,
                    gate_fitness=args.gate_fitness,
                    input_resample_mode=args.input_resample,
                    do_comp_filter=args.legacy_comp_filter,
                    do_reg_filter=not args.no_reg_filter,
                    reg_filter_cfg=rcfg,
                )
            except Exception as e:
                n_skip += 1
                print(f"\n[skip] {stem}: {e}", file=sys.stderr)
                continue
            cname = english_name_from_stem(stem) or "unknown"
            rows.append({
                "class": cname,
                "stem": stem,
                "icp_fitness": out["icp_fitness"],
                "icp_inlier_rmse": out["icp_inlier_rmse"],
                "gate_accepted": out["gate_accepted"],
                "gate_fitness": out["gate_fitness"],
            })

        print(f"\n完成: success={len(rows)}  skip={n_skip}")
        print_batch_summary(rows)
        return

    if not args.stem:
        print(
            "请指定单个样本 ``--stem <无后缀文件名>``，或使用 ``--eval-all`` 或 ``--list-stems``。\n"
            f"列出可用 stem: python scripts/05_estimate_pose.py --data-root {args.data_root} "
            f"--split {args.split} --list-stems",
            file=sys.stderr,
        )
        sys.exit(2)

    split_root = os.path.join(args.data_root, args.split)
    input_path = os.path.join(split_root, "input", f"{args.stem}.npy")
    obs_path = os.path.join(split_root, "obs_w", f"{args.stem}.npy")
    meta_path = os.path.join(split_root, "meta", f"{args.stem}.npz")
    for p in (input_path, obs_path, meta_path, args.ckpt):
        if not os.path.exists(p):
            raise FileNotFoundError(p)

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
        fpfh_voxel=args.fpfh_voxel,
        gate_fitness=args.gate_fitness,
        input_resample_mode=args.input_resample,
        do_comp_filter=args.legacy_comp_filter,
        do_reg_filter=not args.no_reg_filter,
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
