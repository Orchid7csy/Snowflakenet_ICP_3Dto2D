#!/usr/bin/env python3
"""
从本地预处理目录按 synset 类随机抽样，在项目根目录下创建独立文件夹。
默认只导出 input/gt 成对文件；默认在 train,val,test 合并池中抽样（也可用 --split test 只抽 test）。

需要 obs_w、meta 时加: --channels input,gt,obs_w,meta（后两者在源中不存在时会汇总跳过）。

示例:
  python scripts/make_pcn_subset.py \\
    --data-root data/processed/PCN_bbox_pca_in2048_gt16384 \\
    --out pcn_subset_k2 \\
    --per-class 2 --seed 0

  只从 test 抽:
    ... --split test --per-class 2
"""
from __future__ import annotations

import argparse
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent

_SYNSET_IN_STEM = re.compile(r"__(?P<slug>[^_]+)_(?P<synset>\d{8})__")

# 部分预处理（如 bbox_pca）可能不生成 obs_w；meta 也可能未保存
_SOFT_CHANNELS = frozenset({"obs_w", "meta"})


def synset_from_stem(stem: str) -> str | None:
    m = _SYNSET_IN_STEM.search(stem)
    return m.group("synset") if m else None


def list_input_stems(data_root: Path, split: str) -> list[str]:
    inp = data_root / split / "input"
    if not inp.is_dir():
        return []
    return sorted(p.stem for p in inp.glob("*.npy") if p.is_file())


def parse_split_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def stem_split_map(data_root: Path, splits: list[str]) -> dict[str, str]:
    """stem -> 所在 split（train/val/test）。"""
    m: dict[str, str] = {}
    for sp in splits:
        for stem in list_input_stems(data_root, sp):
            m[stem] = sp
    return m


def select_stems_per_class(stems: list[str], per_class: int, seed: int) -> list[str]:
    by_syn: dict[str, list[str]] = defaultdict(list)
    for s in stems:
        sid = synset_from_stem(s)
        if sid is None:
            by_syn["__unknown__"].append(s)
        else:
            by_syn[sid].append(s)
    rng = random.Random(seed)
    chosen: list[str] = []
    for syn in sorted(by_syn.keys()):
        items = by_syn[syn][:]
        rng.shuffle(items)
        k = per_class if per_class > 0 else len(items)
        chosen.extend(items[: min(k, len(items))])
    chosen.sort()
    return chosen


def link_or_copy(src: Path, dst: Path, *, use_symlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if use_symlink:
        os.symlink(src.resolve(), dst, target_is_directory=False)
    else:
        shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="预处理 PCN 按类抽样：默认 input+gt，默认在 train,val,test 合并池中抽样"
    )
    ap.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="预处理输出根目录（含 split/input|gt|...）",
    )
    ap.add_argument(
        "--out",
        type=str,
        default="pcn_processed_subset",
        help="输出文件夹名（建在项目根目录下；也可用绝对路径）",
    )
    ap.add_argument(
        "--split",
        default="train,val,test",
        help="单个 split 或逗号分隔多个（默认 train,val,test 合并后按类抽样）",
    )
    ap.add_argument(
        "--per-class",
        type=int,
        default=3,
        help="每类最多几条；0 表示该类全保留",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--channels",
        type=str,
        default="input,gt",
        help="逗号分隔；默认 input,gt；obs_w/meta 有则拷贝、无则跳过",
    )
    ap.add_argument(
        "--symlink",
        action="store_true",
        help="用符号链接代替拷贝（省空间，勿移动源文件）",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将导出的 stem",
    )
    args = ap.parse_args()

    data_root = Path(args.data_root).expanduser()
    if not data_root.is_absolute():
        data_root = (_PROJECT_ROOT / data_root).resolve()
    else:
        data_root = data_root.resolve()

    out_root = Path(args.out).expanduser()
    if not out_root.is_absolute():
        out_root = (_PROJECT_ROOT / out_root).resolve()
    else:
        out_root = out_root.resolve()

    channels = [c.strip().lower() for c in args.channels.split(",") if c.strip()]
    valid = {"gt", "input", "obs_w", "meta"}
    bad = set(channels) - valid
    if bad:
        print(f"未知 channels: {bad}，允许 {sorted(valid)}", file=sys.stderr)
        return 2

    splits = parse_split_list(args.split)
    st_map = stem_split_map(data_root, splits)
    stems = sorted(st_map.keys())
    if not stems:
        print(
            f"未找到 input: 在 {data_root} 下尝试 splits={splits}",
            file=sys.stderr,
        )
        return 1

    chosen = select_stems_per_class(stems, args.per_class, args.seed)
    n_cls = len({synset_from_stem(s) or "__unknown__" for s in stems})
    cap = "全部" if args.per_class <= 0 else str(args.per_class)
    print(
        f"splits={splits} 合并共 {len(stems)} 条 input，{n_cls} 类，"
        f"抽取 {len(chosen)} 条（每类最多 {cap}），channels={channels} → {out_root}"
    )

    if args.dry_run:
        for s in chosen:
            print(f"{st_map[s]}\t{s}")
        return 0

    subdir_map = {
        "input": "input",
        "gt": "gt",
        "obs_w": "obs_w",
        "meta": "meta",
    }
    ext_map = {"input": ".npy", "gt": ".npy", "obs_w": ".npy", "meta": ".npz"}

    skipped_soft: dict[str, int] = {c: 0 for c in _SOFT_CHANNELS if c in channels}

    for stem in chosen:
        sp = st_map[stem]
        for ch in channels:
            sub = subdir_map[ch]
            ext = ext_map[ch]
            src = data_root / sp / sub / f"{stem}{ext}"
            if not src.is_file():
                if ch in _SOFT_CHANNELS:
                    skipped_soft[ch] = skipped_soft.get(ch, 0) + 1
                    continue
                print(f"[err] 缺少 {ch}: {src}", file=sys.stderr)
                return 1
            dst = out_root / sp / sub / f"{stem}{ext}"
            link_or_copy(src, dst, use_symlink=args.symlink)

    for ch, n in skipped_soft.items():
        if n > 0:
            print(
                f"[note] {ch}: 源中缺失 {n}/{len(chosen)} 个文件，已跳过"
                f"（若使用 bbox_pca 等未生成 {ch} 的预处理，属正常）",
                file=sys.stderr,
            )

    manifest = out_root / "subset_manifest.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest, "w", encoding="utf-8") as f:
        f.write(f"data_root={data_root}\nout_root={out_root}\n")
        f.write(f"splits={splits} per_class={args.per_class} seed={args.seed}\n")
        f.write(f"channels={channels} symlink={args.symlink}\n")
        for s in chosen:
            f.write(f"{st_map[s]}\t{s}\n")
    print(f"清单: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
