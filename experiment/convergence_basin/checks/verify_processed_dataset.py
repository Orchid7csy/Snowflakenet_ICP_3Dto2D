#!/usr/bin/env python3
"""
核查预处理产物（00_preprocess_pcn.py）是否满足「收敛域扩张」管线契约。

检查项：
  - input/*.npy 形状严格为 (2048, 3)，dtype 可 float32/float64
  - gt/*.npy 形状严格为 (16384, 3)（与默认 --num-gt 一致时可核对）
  - obs_w/*.npy 形状严格为 (2048, 3)
  - meta/*.npz 含 centroid_cano/C_cano、scale_cano、view_idx；可选校验 R_far、t_far、T_far_4x4
  - view_idx ∈ [0, 7]（固定 8 视角策略）

用法::
  PYTHONPATH=<项目根> python experiment/convergence_basin/checks/verify_processed_dataset.py \\
      --processed-root data/processed/PCN_far8_cano_in2048_gt16384 --split test

退出码：0 全部通过；1 存在违规或缺失文件。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def _fail(msg: str) -> None:
    print(f"[FAIL] {msg}", file=sys.stderr)


def main() -> int:
    ap = argparse.ArgumentParser(description="核查 canonical 预处理目录契约")
    ap.add_argument("--processed-root", required=True, help="含 train|val|test/{input,gt,obs_w,meta} 的根目录")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--num-input", type=int, default=2048)
    ap.add_argument("--num-gt", type=int, default=16384)
    ap.add_argument("--max-stems", type=int, default=0, help="仅抽查前 N 个 stem（0=全部）")
    ap.add_argument("--strict-meta-keys", action="store_true", help="要求 meta 含 R_far、t_far、T_far_4x4")
    args = ap.parse_args()

    root = Path(args.processed_root).resolve()
    split_dir = root / args.split
    sub = {k: split_dir / k for k in ("input", "gt", "obs_w", "meta")}
    for name, p in sub.items():
        if not p.is_dir():
            _fail(f"缺少目录: {p}（需要 {name}/）")
            return 1

    stems = sorted(x.stem for x in sub["input"].glob("*.npy"))
    if not stems:
        _fail(f"{sub['input']} 下无 .npy")
        return 1
    if args.max_stems > 0:
        stems = stems[: args.max_stems]

    import numpy as np

    n_bad = 0
    for stem in stems:
        ip = sub["input"] / f"{stem}.npy"
        gp = sub["gt"] / f"{stem}.npy"
        op = sub["obs_w"] / f"{stem}.npy"
        mp = sub["meta"] / f"{stem}.npz"
        for label, path in (("input", ip), ("gt", gp), ("obs_w", op)):
            if not path.is_file():
                _fail(f"缺失 {label}: {path}")
                n_bad += 1
                continue
        if n_bad:
            continue

        inp = np.load(ip)
        gt = np.load(gp)
        obs = np.load(op)
        if inp.shape != (args.num_input, 3):
            _fail(f"{stem} input 形状 {inp.shape} != ({args.num_input}, 3)")
            n_bad += 1
        if gt.shape != (args.num_gt, 3):
            _fail(f"{stem} gt 形状 {gt.shape} != ({args.num_gt}, 3)")
            n_bad += 1
        if obs.shape != (args.num_input, 3):
            _fail(f"{stem} obs_w 形状 {obs.shape} != ({args.num_input}, 3)")
            n_bad += 1

        if not mp.is_file():
            _fail(f"缺失 meta: {mp}")
            n_bad += 1
            continue
        z = dict(np.load(mp, allow_pickle=True))
        c = z.get("centroid_cano")
        if c is None:
            c = z.get("C_cano")
        if c is None:
            _fail(f"{stem} meta 无 centroid_cano / C_cano")
            n_bad += 1
        if "scale_cano" not in z:
            _fail(f"{stem} meta 无 scale_cano")
            n_bad += 1
        vi = z.get("view_idx")
        if vi is None:
            _fail(f"{stem} meta 无 view_idx")
            n_bad += 1
        else:
            v = int(np.asarray(vi).ravel()[0])
            if not (0 <= v <= 7):
                _fail(f"{stem} view_idx={v} 不在 [0,7]")
                n_bad += 1
        if args.strict_meta_keys:
            for k in ("R_far", "t_far", "T_far_4x4"):
                if k not in z:
                    _fail(f"{stem} meta 缺少 {k}")
                    n_bad += 1

    if n_bad:
        print(f"核查结束：发现 {n_bad} 个问题。", file=sys.stderr)
        return 1

    print(
        f"[OK] processed-root={root} split={args.split}  "
        f"抽查/全量 stems={len(stems)} | "
        f"input/gt/obs_w/meta 契约满足（input={args.num_input}, gt={args.num_gt}）。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
