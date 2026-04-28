"""
在预处理 .npy 上评测 SnowflakeNet CD-L1×10³（与论文表 1 口径一致）。
"""
from __future__ import annotations

import argparse
import os
import random
import sys
from collections import defaultdict
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_SNET_ROOT = os.path.join(_PROJECT_ROOT, "Snet", "SnowflakeNet-main")
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


def main() -> None:
    ap = argparse.ArgumentParser(description="PCN 预处理 .npy 上 CD-L1×1e3")
    ap.add_argument("--processed-root", required=True)
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument(
        "--ckpt",
        default=os.path.join(
            _SNET_ROOT, "completion", "checkpoints", "ckpt-best-pcn-cd_l1.pth",
        ),
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
    ap.add_argument("--print-paper", action="store_true")
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    processed_root = os.path.abspath(args.processed_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    do_filter = args.use_comp_filter
    model = load_snowflakenet(args.ckpt)
    print(f"设备: {device}  权重: {args.ckpt}")

    if args.print_paper:
        print("Table 1 论文 Ours (SnowflakeNet) CD-L1 x1e3:")
        for k in [
            "Average", "Plane", "Cabinet", "Car", "Chair",
            "Lamp", "Couch", "Table", "Boat",
        ]:
            print(f"  {k:8s}  {PAPER_SNOWFLAKE[k]:.2f}")
        print()

    pairs = list_npy_pairs(processed_root, args.split)
    if not pairs:
        print(f"未找到成对 .npy：{processed_root}/{args.split}/input|gt", file=sys.stderr)
        sys.exit(1)

    pairs = select_per_class(pairs, args.per_class, args.seed)
    print(f"样本数: {len(pairs)}")

    by_class: Dict[str, List[float]] = defaultdict(list)
    all_cds: List[float] = []

    for stem, ip, gp in tqdm(pairs, desc="eval cd-l1"):
        try:
            part = np.load(ip)
            gt = np.load(gp)
            pred = forward_from_npy(
                model,
                part,
                device,
                input_mode=args.input_mode,
                do_comp_filter=do_filter,
                n_input_points=args.n_input_points,
            )
            m = cdl1_times_1e3(pred, gt, device)
        except Exception as e:
            print(f"\n[skip] {stem}: {e}", file=sys.stderr)
            continue
        cname = english_name_from_stem(stem)
        if cname:
            by_class[cname].append(m)
        all_cds.append(m)

    print("\n" + "=" * 72)
    print(f"PCN 预处理 .npy  {args.split}  CD-L1 x1e3  (input_mode={args.input_mode})")
    print("-" * 72)
    print(f"  {'Class':<10s}  {'mean':>8s}  {'n':>4s}  (paper SNet)")
    class_means: List[float] = []
    for _tid, cname in TAXONOMY_TABLE1:
        vals = by_class.get(cname, [])
        if not vals:
            print(
                f"  {cname:<10s}  {'--':>8s}  {0:>4d}  "
                f"({PAPER_SNOWFLAKE.get(cname, 0):.2f})"
            )
            continue
        bar = float(np.mean(vals))
        class_means.append(bar)
        print(
            f"  {cname:<10s}  {bar:8.2f}  {len(vals):4d}  "
            f"({PAPER_SNOWFLAKE.get(cname, 0):.2f})"
        )

    macro = float(np.mean(class_means)) if class_means else float("nan")
    micro = float(np.mean(all_cds)) if all_cds else float("nan")
    print("-" * 72)
    print(f"  {'MacroAvg':<10s}  {macro:8.2f}  paper: {PAPER_SNOWFLAKE['Average']:.2f}")
    print(f"  {'MicroAvg':<10s}  {micro:8.2f}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
