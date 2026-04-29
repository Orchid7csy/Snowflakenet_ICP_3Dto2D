"""
在预处理 .npy 上评测 SnowflakeNet CD-L1×10³（与论文表 1 同一度量口径）。

默认用途：在同一套 `--processed-root` 数据上对比 **官方预训练权重** 与 **本文微调权重**，
不再默认附带「论文在原始 PCN benchmark 上的数值」列。
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_SNET_ROOT = os.path.join(_PROJECT_ROOT, "Snet", "SnowflakeNet-main")
_DEFAULT_OFFICIAL_CKPT = os.path.join(
    _SNET_ROOT, "completion", "checkpoints", "ckpt-best-pcn-cd_l1.pth",
)
for p in (_PROJECT_ROOT, _SNET_ROOT, _SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.evaluation.cd_l1 import (  # noqa: E402
    PAPER_SNOWFLAKE,
    TAXONOMY_TABLE1,
    cdl1_times_1e3,
    english_name_from_stem,
    list_npy_pairs,
    select_per_class,
)
from src.evaluation.npy_forward import forward_from_npy  # noqa: E402
from src.models.snet_loader import load_snowflakenet  # noqa: E402


@dataclass
class EvalStats:
    """各类 mean CD-L1×10³ + Macro/Micro（与表 1 度量一致，数据可为任意预处理）。"""

    class_rows: List[Tuple[str, float, int]]  # (english_name, mean, n)
    macro: float
    micro: float


def _evaluate_one_ckpt(
    ckpt_path: str,
    pairs: List[Tuple[str, str, str]],
    device: torch.device,
    *,
    input_mode: str,
    do_filter: bool,
    n_input_points: int,
    desc: str,
) -> EvalStats:
    model = load_snowflakenet(ckpt_path)
    by_class: Dict[str, List[float]] = defaultdict(list)
    all_cds: List[float] = []

    for stem, ip, gp in tqdm(pairs, desc=desc):
        try:
            part = np.load(ip)
            gt = np.load(gp)
            pred = forward_from_npy(
                model,
                part,
                device,
                input_mode=input_mode,
                do_comp_filter=do_filter,
                n_input_points=n_input_points,
            )
            m = cdl1_times_1e3(pred, gt, device)
        except Exception as e:
            print(f"\n[skip] {stem}: {e}", file=sys.stderr)
            continue
        cname = english_name_from_stem(stem)
        if cname:
            by_class[cname].append(m)
        all_cds.append(m)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    class_rows: List[Tuple[str, float, int]] = []
    class_means: List[float] = []
    for _tid, cname in TAXONOMY_TABLE1:
        vals = by_class.get(cname, [])
        if not vals:
            class_rows.append((cname, float("nan"), 0))
            continue
        bar = float(np.mean(vals))
        class_means.append(bar)
        class_rows.append((cname, bar, len(vals)))

    macro = float(np.mean(class_means)) if class_means else float("nan")
    micro = float(np.mean(all_cds)) if all_cds else float("nan")
    return EvalStats(class_rows=class_rows, macro=macro, micro=micro)


def _print_compare_table(
    stats_official: EvalStats,
    stats_ours: EvalStats,
    *,
    split: str,
    input_mode: str,
    label_official: str,
    label_ours: str,
) -> None:
    w = 72
    print("\n" + "=" * w)
    print(
        f"同一预处理 .npy   split={split}   CD-L1×10³   (input_mode={input_mode})\n"
        "列为「官方预训练 SnowflakeNet」与「本文微调」在同一数据上的对比（非论文原始 PCN 测试协议）。"
    )
    print("-" * w)
    hdr = (
        f"  {'Class':<10s}  "
        f"{label_official[:14]:>14s}  "
        f"{label_ours[:14]:>14s}  "
        f"{'n':>4s}"
    )
    print(hdr)
    for (c_o, m_o, n_o), (c_u, m_u, n_u) in zip(
        stats_official.class_rows, stats_ours.class_rows
    ):
        assert c_o == c_u
        so = f"{m_o:8.2f}" if np.isfinite(m_o) else "      --"
        su = f"{m_u:8.2f}" if np.isfinite(m_u) else "      --"
        n = n_o if n_o == n_u else max(n_o, n_u)
        print(f"  {c_o:<10s}  {so:>14s}  {su:>14s}  {n:4d}")

    print("-" * w)
    mo = (
        f"{stats_official.macro:8.2f}"
        if np.isfinite(stats_official.macro)
        else "      --"
    )
    mu = (
        f"{stats_ours.macro:8.2f}"
        if np.isfinite(stats_ours.macro)
        else "      --"
    )
    print(f"  {'MacroAvg':<10s}  {mo:>14s}  {mu:>14s}")
    mo2 = (
        f"{stats_official.micro:8.2f}"
        if np.isfinite(stats_official.micro)
        else "      --"
    )
    mu2 = (
        f"{stats_ours.micro:8.2f}"
        if np.isfinite(stats_ours.micro)
        else "      --"
    )
    print(f"  {'MicroAvg':<10s}  {mo2:>14s}  {mu2:>14s}")
    print("=" * w + "\n")


def _print_single_table(
    stats: EvalStats,
    *,
    split: str,
    input_mode: str,
    label: str,
) -> None:
    w = 72
    print("\n" + "=" * w)
    print(
        f"预处理 .npy   split={split}   CD-L1×10³   (input_mode={input_mode})\n"
        f"模型: {label}"
    )
    print("-" * w)
    print(f"  {'Class':<10s}  {'mean':>8s}  {'n':>4s}")
    for cname, bar, n in stats.class_rows:
        if not np.isfinite(bar):
            print(f"  {cname:<10s}  {'--':>8s}  {n:4d}")
        else:
            print(f"  {cname:<10s}  {bar:8.2f}  {n:4d}")
    print("-" * w)
    ma = f"{stats.macro:8.2f}" if np.isfinite(stats.macro) else "nan"
    mi = f"{stats.micro:8.2f}" if np.isfinite(stats.micro) else "nan"
    print(f"  {'MacroAvg':<10s}  {ma:>8s}")
    print(f"  {'MicroAvg':<10s}  {mi:>8s}")
    print("=" * w + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "预处理 .npy 上 CD-L1×10³。指定 --ours-ckpt 时与官方预训练在同一数据上对比；"
            "否则仅评单个 --ckpt（默认官方权重）。"
        )
    )
    ap.add_argument("--processed-root", required=True)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument(
        "--official-ckpt",
        default=_DEFAULT_OFFICIAL_CKPT,
        help="SnowflakeNet 官方 PCN 预训练权重（对比模式左列）",
    )
    ap.add_argument(
        "--ours-ckpt",
        default=None,
        help="本文微调 checkpoint；指定后与 --official-ckpt 并排对比",
    )
    ap.add_argument(
        "--ckpt",
        default=None,
        help="仅单模型评测时使用（旧参数）；与 --ours-ckpt 互斥",
    )
    ap.add_argument("--per-class", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--input-mode",
        choices=("direct", "upsample", "legacy"),
        default="direct",
    )
    ap.add_argument("--use-comp-filter", action="store_true")
    ap.add_argument("--n-input-points", type=int, default=2048)
    ap.add_argument(
        "--print-paper",
        action="store_true",
        help="额外打印论文 Table 1 报道值（原始 PCN benchmark，便于脚注对照）",
    )
    args = ap.parse_args()

    if args.ckpt and args.ours_ckpt:
        print("请只指定 --ckpt 或 --ours-ckpt 之一。", file=sys.stderr)
        sys.exit(2)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    processed_root = os.path.abspath(args.processed_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    do_filter = args.use_comp_filter

    pairs = list_npy_pairs(processed_root, args.split)
    if not pairs:
        print(f"未找到成对 .npy：{processed_root}/{args.split}/input|gt", file=sys.stderr)
        sys.exit(1)

    pairs = select_per_class(pairs, args.per_class, args.seed)
    print(f"设备: {device}  样本数: {len(pairs)}")

    if args.print_paper:
        print("Table 1 论文 SnowflakeNet CD-L1×10³（原始 PCN benchmark，仅供参考）：")
        for k in [
            "Average", "Plane", "Cabinet", "Car", "Chair",
            "Lamp", "Couch", "Table", "Boat",
        ]:
            print(f"  {k:8s}  {PAPER_SNOWFLAKE[k]:.2f}")
        print()

    if args.ours_ckpt:
        print(f"对比 | 官方预训练: {args.official_ckpt}")
        print(f"     | 本文微调:    {args.ours_ckpt}")
        stats_o = _evaluate_one_ckpt(
            args.official_ckpt,
            pairs,
            device,
            input_mode=args.input_mode,
            do_filter=do_filter,
            n_input_points=args.n_input_points,
            desc="official cd-l1",
        )
        stats_u = _evaluate_one_ckpt(
            args.ours_ckpt,
            pairs,
            device,
            input_mode=args.input_mode,
            do_filter=do_filter,
            n_input_points=args.n_input_points,
            desc="ours cd-l1",
        )
        _print_compare_table(
            stats_o,
            stats_u,
            split=args.split,
            input_mode=args.input_mode,
            label_official="official",
            label_ours="ours (finetuned)",
        )
        return

    single_ckpt = args.ckpt if args.ckpt else _DEFAULT_OFFICIAL_CKPT
    print(f"单模型评测  权重: {single_ckpt}")
    stats = _evaluate_one_ckpt(
        single_ckpt,
        pairs,
        device,
        input_mode=args.input_mode,
        do_filter=do_filter,
        n_input_points=args.n_input_points,
        desc="eval cd-l1",
    )
    _print_single_table(
        stats,
        split=args.split,
        input_mode=args.input_mode,
        label=single_ckpt,
    )


if __name__ == "__main__":
    main()
