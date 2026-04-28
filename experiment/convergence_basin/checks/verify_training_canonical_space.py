#!/usr/bin/env python3
"""
说明性核查：微调脚本使用的 ``CompletionDataset`` 仅从 ``{split}/input`` 与 ``{split}/gt``
读取 **canonical** 空间下的 .npy；损失在 tensor 空间计算，不包含 FPFH/T_far 位姿。

可选：加载单个 batch 打印形状（需 PyTorch）。

用法::
  PYTHONPATH=<项目根> python experiment/convergence_basin/checks/verify_training_canonical_space.py \\
      --data-root data/processed/PCN_far8_cano_in2048_gt16384 --split train --dry-run

``--dry-run``（默认）：仅打印结论与路径；不加则尝试 ``DataLoader`` 取一个 batch。
"""
from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--dry-run", action="store_true", help="不实例化 DataLoader")
    args = ap.parse_args()

    root = os.path.abspath(args.data_root)
    inp = os.path.join(root, args.split, "input")
    gt = os.path.join(root, args.split, "gt")
    if not os.path.isdir(inp) or not os.path.isdir(gt):
        print(f"[FAIL] 缺少 input/gt: {inp} / {gt}", file=sys.stderr)
        return 1

    print(
        "[INFO] 训练损失仅在 canonical 坐标下计算：\n"
        "  - input、gt 来自预处理 normalize_by_complete + 保存的张量；\n"
        "  - obs_w、meta 不参与 CompletionDataset 前向；\n"
        "  - PCNRotAugCompletionDataset 仅在 canonical 内做 SO(3)/符号旋转增强。\n"
        f"  data-root={root} split={args.split}"
    )

    if args.dry_run:
        print("[OK] dry-run：未加载 DataLoader。")
        return 0

    from torch.utils.data import DataLoader

    from src.data.pcn_dataset import CompletionDataset

    ds = CompletionDataset(root, split=args.split, input_points=2048, gt_points=16384)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    inp_b, gt_b = next(iter(loader))
    print(f"[OK] 样本 batch 形状: input={tuple(inp_b.shape)} gt={tuple(gt_b.shape)}（应为 (B,2048,3) 与 (B,16384,3)）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
