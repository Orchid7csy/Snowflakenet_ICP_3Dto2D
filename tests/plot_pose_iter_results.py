#!/usr/bin/env python3
"""
从 ``pose_sample_details.csv`` / ``pose_iter_summary.csv`` 生成位姿迭代评估图。

与 ``scripts/05_estimate_pose.py`` 中 ``_finalize_pose_logging_to_csv`` 的列类型一致：
  - 布尔 ``gate_passed``、整数 ``final_iter_count``、浮点 metrics。
  - 扁平行仅保留 iter_1..3 的 Gate fitness；``final_iter_count`` 可大于 3。

用法示例（在项目根 ``SnowflakeNet_FPFH_ICP/`` 下）::

    # 默认读取仓库根目录下的 pose_sample_details.csv、pose_iter_summary.csv，
    # PNG 输出到 figures/pose_iter/
    python3 tests/plot_pose_iter_results.py

    # 指定 CSV 与输出目录（例如 05 写了 results/ 时）
    python3 tests/plot_pose_iter_results.py \\
        --details results/pose_sample_details.csv \\
        --summary results/pose_iter_summary.csv \\
        --out figures/pose_iter_run2

    # 仅有 details、不需要汇总条形图时（显式跳过 summary）
    python3 tests/plot_pose_iter_results.py --details ./pose_sample_details.csv --no-summary

    # 若未传 --no-summary 且 --summary 指向的文件不存在，会自动跳过第四张汇总图并打印提示

    # 更高分辨率导出
    python3 tests/plot_pose_iter_results.py --dpi 300 --out figures/pose_iter_hd

生成文件（默认 ``--out figures/pose_iter``）::

    pose_iter_count_and_gate_rate.png   # final_iter_count 直方图 + 分层 Gate 通过率
    pose_base_vs_ours_fitness.png       # base vs ours_final fitness 散点
    pose_class_gate_pass_topn.png      # 样本数最多的若干类的 Gate 通过率
    pose_iter_summary_percentages.png  # 需有效 summary：汇总 CSV 中的百分比横条

依赖: ``numpy``, ``matplotlib``（不设 pandas 硬依赖；与 ``scripts/05_estimate_pose.py`` 写入的 CSV 列兼容）。
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Any

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as e:  # pragma: no cover
    raise SystemExit("需要 matplotlib: pip install matplotlib") from e


def _repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _parse_float(s: str) -> float:
    s = (s or "").strip()
    if not s:
        return math.nan
    return float(s)


def _parse_int(s: str) -> int:
    return int((s or "").strip())


def _parse_bool(s: str) -> bool:
    return (s or "").strip().lower() == "true"


def load_details(path: str) -> dict[str, np.ndarray]:
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    n = len(rows)
    out: dict[str, Any] = {
        "sample_id": np.array([row["sample_id"] for row in rows], dtype=object),
        "class_name": np.array([row["class_name"] for row in rows], dtype=object),
        "base_fitness": np.empty(n, dtype=np.float64),
        "base_rmse": np.empty(n, dtype=np.float64),
        "final_iter_count": np.empty(n, dtype=np.int32),
        "gate_passed": np.empty(n, dtype=bool),
        "iter_1_fitness": np.empty(n, dtype=np.float64),
        "iter_2_fitness": np.empty(n, dtype=np.float64),
        "iter_3_fitness": np.empty(n, dtype=np.float64),
        "ours_final_fitness": np.empty(n, dtype=np.float64),
        "ours_final_rmse": np.empty(n, dtype=np.float64),
    }
    for i, row in enumerate(rows):
        out["base_fitness"][i] = _parse_float(row["base_fitness"])
        out["base_rmse"][i] = _parse_float(row["base_rmse"])
        out["final_iter_count"][i] = _parse_int(row["final_iter_count"])
        out["gate_passed"][i] = _parse_bool(row["gate_passed"])
        out["iter_1_fitness"][i] = _parse_float(row["iter_1_fitness"])
        out["iter_2_fitness"][i] = _parse_float(row["iter_2_fitness"])
        out["iter_3_fitness"][i] = _parse_float(row["iter_3_fitness"])
        out["ours_final_fitness"][i] = _parse_float(row["ours_final_fitness"])
        out["ours_final_rmse"][i] = _parse_float(row["ours_final_rmse"])
    return out


def load_summary(path: str) -> dict[str, str]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"预期 summary 一行表头 + 一行数据，得到 {len(rows)} 行数据")
    return {k: rows[0][k] for k in rows[0]}


def _fig_final_iter_and_gate(d: dict[str, np.ndarray], ax_bar: Any, ax_frac: Any) -> None:
    ic = d["final_iter_count"]
    gp = d["gate_passed"]
    counts = Counter(int(x) for x in ic.tolist())
    k_max = max(counts) if counts else 0
    xs = list(range(1, k_max + 1))
    ys = [counts.get(k, 0) for k in xs]
    ax_bar.bar(xs, ys, color="steelblue", edgecolor="black", linewidth=0.5)
    ax_bar.set_xlabel("final_iter_count")
    ax_bar.set_ylabel("count")
    ax_bar.set_title("Stop iteration (early exit on Gate OK / ICP converge)")
    ax_bar.set_xticks(xs)

    passed = np.zeros(len(xs), dtype=np.float64)
    total = np.zeros(len(xs), dtype=np.float64)
    for i, k in enumerate(xs):
        m = ic == k
        total[i] = float(m.sum())
        passed[i] = float((m & gp).sum())
    frac = np.divide(passed, total, out=np.zeros_like(passed), where=total > 0)
    ax_frac.bar(xs, 100.0 * frac, color="seagreen", edgecolor="black", linewidth=0.5)
    ax_frac.set_xlabel("final_iter_count")
    ax_frac.set_ylabel("gate_passed rate within bucket (%)")
    ax_frac.set_title("Gate pass rate by final_iter_count")
    ax_frac.set_xticks(xs)
    ax_frac.set_ylim(0, 105)


def _fig_summary_bars(summary: dict[str, str], ax: Any) -> None:
    keys = [
        ("pct_pass_gate_when_iter_eq_1", "pass & stop at iter==1"),
        ("pct_pass_gate_when_iter_eq_2", "pass & stop at iter==2"),
        ("pct_pass_gate_when_iter_eq_3", "pass & stop at iter==3"),
        ("pct_fallback_timeout_no_gate", "TIMEOUT, never passed gate"),
    ]
    labels = [b for _, b in keys]
    vals = [float(summary[k]) for k, _ in keys]
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, vals, color="coral", edgecolor="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("% of all samples")
    ax.set_title(
        f'Summary (n={summary["n_samples"]}, '
        f'configured_max_iter={summary["configured_max_iter"]}, '
        f'iterative_refine={summary["iterative_refine_flag"]})'
    )
    for i, v in enumerate(vals):
        ax.text(v + 0.3, i, f"{v:.2f}%", va="center", fontsize=9)


def _fig_scatter_base_vs_ours(d: dict[str, np.ndarray], ax: Any) -> None:
    bf = d["base_fitness"]
    of = d["ours_final_fitness"]
    gp = d["gate_passed"]
    ax.scatter(bf[~gp], of[~gp], s=8, alpha=0.5, c="tab:red", label="gate_passed=False")
    ax.scatter(bf[gp], of[gp], s=8, alpha=0.5, c="tab:blue", label="gate_passed=True")
    lim = max(float(np.nanmax(bf)), float(np.nanmax(of)), 1e-6)
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("base_fitness")
    ax.set_ylabel("ours_final_fitness")
    ax.set_title("Final ICP fitness vs baseline (diagonal = no change)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_aspect("equal", adjustable="box")


def _fig_class_pass_rate(d: dict[str, np.ndarray], ax: Any, max_classes: int = 16) -> None:
    classes = d["class_name"]
    gp = d["gate_passed"]
    by: dict[str, list[bool]] = defaultdict(list)
    for c, g in zip(classes.tolist(), gp.tolist()):
        by[str(c)].append(bool(g))
    rates = sorted(
        ((c, sum(gs) / len(gs), len(gs)) for c, gs in by.items()),
        key=lambda t: -t[2],
    )[:max_classes]
    if not rates:
        ax.set_visible(False)
        return
    labels = [f"{c} (n={n})" for c, _p, n in rates]
    vals = [100.0 * p for c, p, n in rates]
    y = np.arange(len(labels))
    ax.barh(y, vals, color="mediumpurple", edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("gate_passed rate (%)")
    ax.set_title(f"Per-class gate pass (top {len(labels)} by count)")
    ax.set_xlim(0, 105)


def main() -> None:
    root = _repo_root()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--details",
        default=os.path.join(root, "pose_sample_details.csv"),
        help="per-sample CSV",
    )
    ap.add_argument(
        "--summary",
        default=os.path.join(root, "pose_iter_summary.csv"),
        help="单行汇总 CSV",
    )
    ap.add_argument(
        "--no-summary",
        action="store_true",
        help="不读取 summary，跳过 pose_iter_summary_percentages.png",
    )
    ap.add_argument(
        "--out",
        default=os.path.join(root, "figures", "pose_iter"),
        help="输出目录（写入 PNG）",
    )
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    if not os.path.isfile(args.details):
        print(f"找不到 details: {args.details}", file=sys.stderr)
        sys.exit(1)
    d = load_details(args.details)
    summary: dict[str, str] | None
    if args.no_summary:
        summary = None
    elif os.path.isfile(args.summary):
        summary = load_summary(args.summary)
    else:
        summary = None
        print(f"未找到 summary，跳过汇总条形图: {args.summary}", file=sys.stderr)

    os.makedirs(args.out, exist_ok=True)

    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    _fig_final_iter_and_gate(d, ax1, ax2)
    fig1.tight_layout()
    p1 = os.path.join(args.out, "pose_iter_count_and_gate_rate.png")
    fig1.savefig(p1, dpi=args.dpi)
    plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(8, 3.8))
    _fig_scatter_base_vs_ours(d, ax)
    fig2.tight_layout()
    p2 = os.path.join(args.out, "pose_base_vs_ours_fitness.png")
    fig2.savefig(p2, dpi=args.dpi)
    plt.close(fig2)

    fig3, ax = plt.subplots(figsize=(7, 5))
    _fig_class_pass_rate(d, ax)
    fig3.tight_layout()
    p3 = os.path.join(args.out, "pose_class_gate_pass_topn.png")
    fig3.savefig(p3, dpi=args.dpi)
    plt.close(fig3)

    if summary is not None:
        fig4, ax = plt.subplots(figsize=(8, 3.5))
        _fig_summary_bars(summary, ax)
        fig4.tight_layout()
        p4 = os.path.join(args.out, "pose_iter_summary_percentages.png")
        fig4.savefig(p4, dpi=args.dpi)
        plt.close(fig4)

    written = [p1, p2, p3]
    if summary is not None:
        written.append(p4)
    print("Wrote:")
    for p in written:
        print(" ", p)


if __name__ == "__main__":
    main()
