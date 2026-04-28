"""
PCN 预处理：对象系 partial/complete 读入 → 随机 T_far(远距) → 在 obs_w 上做 bbox 归一化 + PCA →
重采样为 input=2048 / gt=16384，并保存 obs_w 与 meta（含 R_far、t_far、C_bbox、scale、R_pca、mu_pca）。

用法:
  python scripts/00_preprocess_pcn.py --pcn-root PCN --splits train,val,test
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import zlib
from pathlib import Path

import numpy as np
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.naming import sample_stem  # noqa: E402
from src.data import preprocessing as prep  # noqa: E402
from src.utils.io import read_pcd_xyz  # noqa: E402


def _view_idx_from_pcd_name(name: str) -> int:
    stem = Path(name).stem
    if re.match(r"^\d+$", stem):
        return int(stem)
    m = re.search(r"(\d+)", stem)
    if m:
        return int(m.group(1))
    return 0


def _iter_pairs(
    pcn_root: Path, splits: list[str]
) -> list[tuple[str, str, str, Path, Path, int]]:
    out: list[tuple[str, str, str, Path, Path, int]] = []
    for split in splits:
        comp_root = pcn_root / split / "complete"
        if not comp_root.is_dir():
            continue
        for tax_dir in sorted(comp_root.iterdir()):
            if not tax_dir.is_dir():
                continue
            synset = tax_dir.name
            for gtp in sorted(tax_dir.glob("*.pcd")):
                model_id = gtp.stem
                pdir = pcn_root / split / "partial" / synset / model_id
                if not pdir.is_dir():
                    continue
                for pp in sorted(pdir.glob("*.pcd")):
                    view_idx = _view_idx_from_pcd_name(pp.name)
                    out.append((split, synset, model_id, pp, gtp, view_idx))
    return out


def _process_one(
    split: str,
    synset: str,
    model_id: str,
    partial_pcd: Path,
    complete_pcd: Path,
    view_idx: int,
    *,
    num_input: int,
    num_gt: int,
    pca_axis: str,
    base_seed: int,
    t_min: float,
    t_max: float,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, dict]:
    stem = sample_stem(split=split, synset=synset, model_id=model_id, view=view_idx)
    rng = np.random.default_rng(
        (base_seed + zlib.adler32(stem.encode("utf-8"))) % (2**32)
    )

    part_obj = read_pcd_xyz(partial_pcd)
    gt_obj = read_pcd_xyz(complete_pcd)
    r_far, t_far, t_norm = prep.sample_random_far_transform(rng, t_min, t_max)
    obs_w = prep.apply_rigid_row(part_obj, r_far, t_far)
    gt_w = prep.apply_rigid_row(gt_obj, r_far, t_far)

    p_norm, c_bbox, scale = prep.normalize_by_bbox(obs_w)
    p_in, r_pca, mu_pca = prep.pca_align(p_norm, target_axis=pca_axis)
    gt_norm = ((gt_w - c_bbox[None, :]) / np.float32(scale)).astype(np.float32)
    gt_out = prep.apply_pca_rigid(gt_norm, r_pca, mu_pca)

    p_rs = prep.resample_rng(p_in, num_input, rng)
    g_rs = prep.resample_rng(gt_out, num_gt, rng)
    obs_rs = prep.resample_rng(obs_w, num_input, rng)
    t_far_4x4 = prep.rigid_T_4x4(r_far, t_far)
    meta = {
        "C_bbox": c_bbox,
        "scale": np.float32(scale),
        "R_pca": r_pca,
        "mu_pca": mu_pca,
        "pca_axis": pca_axis,
        "split": split,
        "synset": synset,
        "model_id": model_id,
        "view_idx": np.int32(view_idx),
        "R_far": r_far,
        "t_far": t_far,
        "t_norm": np.float32(t_norm),
        "T_far_4x4": t_far_4x4,
        "source_partial": str(partial_pcd),
        "source_complete": str(complete_pcd),
    }
    return stem, p_rs, g_rs, obs_rs, meta


def main() -> int:
    ap = argparse.ArgumentParser(description="PCN：T_far + bbox+PCA → input/gt/obs_w + meta")
    ap.add_argument("--pcn-root", type=str, default=os.path.join(_PROJECT_ROOT, "PCN"))
    ap.add_argument(
        "--out-root",
        type=str,
        default="",
        help="默认: data/processed/PCN_far_in{input}_gt{gt}/",
    )
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--num-input", type=int, default=2048)
    ap.add_argument("--num-gt", type=int, default=16384)
    ap.add_argument("--pca-axis", default="z", choices=("x", "y", "z"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t-min", type=float, default=1.0, help="t_far 模长下界")
    ap.add_argument("--t-max", type=float, default=3.0, help="t_far 模长上界")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--no-meta", action="store_true")
    args = ap.parse_args()

    pcn_root = Path(args.pcn_root).resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not args.out_root:
        out_root = Path(
            _PROJECT_ROOT, "data", "processed", f"PCN_far_in{args.num_input}_gt{args.num_gt}"
        )
    else:
        out_root = Path(args.out_root).resolve()

    pairs = _iter_pairs(pcn_root, splits)
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]

    if not pairs:
        print("无 (partial, complete) 对", file=sys.stderr)
        return 1

    n_ok, n_skip, n_fail = 0, 0, 0
    for item in tqdm(pairs, desc="pcn T_far+bbox+pca"):
        split, synset, model_id, pp, gp, view_idx = item
        try:
            stem, pin, gto, obw, meta = _process_one(
                split, synset, model_id, pp, gp, view_idx,
                num_input=args.num_input,
                num_gt=args.num_gt,
                pca_axis=args.pca_axis,
                base_seed=args.seed,
                t_min=args.t_min,
                t_max=args.t_max,
            )
        except Exception as e:
            n_fail += 1
            print(f"\n[fail] {pp}: {e}", file=sys.stderr)
            continue

        idir = out_root / split / "input"
        gdir = out_root / split / "gt"
        odir = out_root / split / "obs_w"
        mdir = out_root / split / "meta"
        for d in (idir, gdir, odir):
            d.mkdir(parents=True, exist_ok=True)
        if not args.no_meta:
            mdir.mkdir(parents=True, exist_ok=True)

        ip = idir / f"{stem}.npy"
        gpout = gdir / f"{stem}.npy"
        op = odir / f"{stem}.npy"
        mp = mdir / f"{stem}.npz"

        if args.skip_existing and ip.is_file() and gpout.is_file() and op.is_file():
            n_skip += 1
            continue
        try:
            np.save(str(ip), pin.astype(np.float32))
            np.save(str(gpout), gto.astype(np.float32))
            np.save(str(op), obw.astype(np.float32))
            if not args.no_meta:
                np.savez(str(mp), **meta)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            print(f"\n[fail] save {stem}: {e}", file=sys.stderr)

    manifest = out_root / "00_preprocess_pcn_manifest.txt"
    with open(manifest, "w", encoding="utf-8") as f:
        f.write(f"pcn_root={pcn_root}\nout_root={out_root}\n")
        f.write(
            f"ok={n_ok} skip={n_skip} fail={n_fail} pairs={len(pairs)} "
            f"input={args.num_input} gt={args.num_gt} t_min={args.t_min} t_max={args.t_max}\n"
        )
    print(f"完成: 写入 {n_ok}, 跳过 {n_skip}, 失败 {n_fail} / {len(pairs)}. 输出: {out_root}\n摘要: {manifest}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
