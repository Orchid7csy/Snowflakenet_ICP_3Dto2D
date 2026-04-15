"""
scripts/inspect_remaining_points.py

遍历预处理后的 input 点云，统计“剩余点数”。

重要说明：
- 你的 preprocessing 会把 input 强制重采样到固定点数（通常 2048）。
- 因此想看“退化后真实剩余点数”，只能用近似指标：统计点云中**唯一点数**。
  当点数不足而用 replace=True 填充时，会产生重复点；unique 数越小，说明原始残缺越极端。

用法：
  python3 scripts/inspect_remaining_points.py \
    --data-root data/processed_with_removal --split test

输出：
  - 终端打印汇总
  - outputs/remaining_points/<tag>.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime
from typing import Tuple

import numpy as np


def _abs_under_project(root: str, p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.abspath(os.path.join(root, p))


def count_unique_points(points: np.ndarray, decimals: int = 6) -> int:
    """
    统计唯一点数（近似）：按给定小数位 round 后去重。
    decimals 越小容忍噪声越大。
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return 0
    key = np.round(pts[:, :3], decimals=decimals)
    uniq = np.unique(key, axis=0)
    return int(uniq.shape[0])


def scan_input_dir(input_dir: str, decimals: int) -> Tuple[list[dict], dict]:
    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".npy")])
    rows = []
    uniq_counts = []
    total_counts = []

    for f in files:
        path = os.path.join(input_dir, f)
        p = np.load(path)
        n = int(p.shape[0]) if isinstance(p, np.ndarray) and p.ndim == 2 else 0
        u = count_unique_points(p, decimals=decimals) if n > 0 else 0
        rows.append({"file": f, "N": n, "unique": u})
        total_counts.append(n)
        uniq_counts.append(u)

    def _summary(arr):
        if not arr:
            return {"min": None, "p50": None, "p90": None, "max": None, "mean": None}
        a = np.array(arr, dtype=np.float64)
        return {
            "min": float(np.min(a)),
            "p50": float(np.percentile(a, 50)),
            "p90": float(np.percentile(a, 90)),
            "max": float(np.max(a)),
            "mean": float(np.mean(a)),
        }

    summary = {
        "num_files": len(files),
        "total_N": _summary(total_counts),
        "unique": _summary(uniq_counts),
    }
    return rows, summary


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    ap = argparse.ArgumentParser(description="统计预处理后 input 点云的剩余点数（unique 近似）")
    ap.add_argument("--data-root", default="data/processed_with_removal", help="包含 train/test 的根目录")
    ap.add_argument("--split", default="test", choices=("train", "test"))
    ap.add_argument("--decimals", type=int, default=6, help="unique 统计时 round 的小数位")
    ap.add_argument("--out-dir", default="outputs/remaining_points")
    args = ap.parse_args()

    data_root = _abs_under_project(project_root, args.data_root)
    input_dir = os.path.join(data_root, args.split, "input")
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(input_dir)

    rows, summary = scan_input_dir(input_dir, decimals=args.decimals)

    out_dir = _abs_under_project(project_root, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(out_dir, f"remaining_points_{os.path.basename(data_root)}_{args.split}_{tag}.csv")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["file", "N", "unique"])
        w.writeheader()
        w.writerows(rows)

    print("=== Remaining Points Report ===")
    print(f"data_root : {data_root}")
    print(f"split     : {args.split}")
    print(f"input_dir : {input_dir}")
    print(f"files     : {summary['num_files']}")
    print(f"total_N   : {summary['total_N']}")
    print(f"unique(~remaining): {summary['unique']}  (decimals={args.decimals})")
    print(f"saved_csv : {out_csv}")


if __name__ == "__main__":
    main()

