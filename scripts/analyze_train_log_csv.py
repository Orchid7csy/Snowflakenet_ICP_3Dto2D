"""
scripts/analyze_train_log_csv.py

读取 scripts/03_train_completion_itodd.py 等写出的 train_log_*.csv，做简要诊断：
- best val 所在 epoch、train/val gap
- 末段 val 的波动（标准差、极差），用于判断「是否主要为验证噪声 / 过拟合」

用法:
  python scripts/analyze_train_log_csv.py /path/to/train_log_20260416_024816.csv
  python scripts/analyze_train_log_csv.py /path/to/train_log_20260416_024816.csv --tail 8
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys


def main() -> None:
    ap = argparse.ArgumentParser(description="分析训练 CSV 日志")
    ap.add_argument("csv_path", help="train_log_*.csv 路径")
    ap.add_argument("--tail", type=int, default=0, help="仅分析最后 N 个 epoch 的 val（0 表示用全部）")
    args = ap.parse_args()

    path = os.path.abspath(args.csv_path)
    if not os.path.isfile(path):
        print(f"错误: 文件不存在: {path}", file=sys.stderr)
        sys.exit(1)

    rows = []
    with open(path, newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    if not rows:
        print("错误: CSV 无数据行", file=sys.stderr)
        sys.exit(1)

    epochs = [int(row["epoch"]) for row in rows]
    train = [float(row["train_loss"]) for row in rows]
    val = [float(row["val_loss"]) for row in rows]
    lr = [float(row["lr"]) for row in rows]

    best_i = int(min(range(len(val)), key=lambda i: val[i]))
    last = rows[-1]
    print(f"文件: {path}")
    print(f"epoch 范围: {epochs[0]}–{epochs[-1]}  (共 {len(rows)} 行)")
    print(f"最优 val: {val[best_i]:.6f} @ epoch {epochs[best_i]}")
    print(f"末行: epoch {last['epoch']} train={float(last['train_loss']):.6f} val={float(last['val_loss']):.6f} lr={last['lr']}")

    gap = [train[i] - val[i] for i in range(len(rows))]
    print(f"train - val: 末 epoch {gap[-1]:.6f}（负表示 val 高于 train，常见于 completion / 难 val）")

    tail_n = args.tail if args.tail and args.tail > 0 else len(val)
    tail_n = min(tail_n, len(val))
    vtail = val[-tail_n:]
    mean_v = sum(vtail) / len(vtail)
    var_v = sum((x - mean_v) ** 2 for x in vtail) / len(vtail)
    std_v = math.sqrt(var_v) if var_v > 0 else 0.0
    print(f"\nval 末 {tail_n} 个 epoch: mean={mean_v:.6f} std={std_v:.6f} min={min(vtail):.6f} max={max(vtail):.6f} range={max(vtail)-min(vtail):.6f}")

    if std_v > 0.03 and tail_n >= 5:
        print(
            "\n提示: 末段 val 波动较大时，单点「最优 epoch」可能偶然；可结合多 seed、"
            "滑动平均 val，或以 test 上单次前向为准。"
        )


if __name__ == "__main__":
    main()
