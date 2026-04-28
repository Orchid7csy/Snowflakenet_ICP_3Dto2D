"""
PCN 最优 HPR 视角：对每个 (split, synset, model_id) 选 partial，使 PCA 对齐后的 complete 在 Fibonacci HPR 下可见点数最大。

用法:
  python scripts/01_compute_best_view.py --pcn-root PCN --splits test --out-json data/processed/PCN_hpr_best_views.json
  python scripts/01_compute_best_view.py --pcn-root PCN --splits train,val,test --workers 32
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import numpy as np
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.data.naming import sample_stem  # noqa: E402
from src.data import preprocessing as _prep  # noqa: E402
from src.data.hpr import hpr_radius_effective, max_hpr_visibility_count  # noqa: E402


def _read_pcd_xyz(path: Path) -> np.ndarray:
    import open3d as o3d

    pc = o3d.io.read_point_cloud(str(path))
    pts = np.asarray(pc.points, dtype=np.float64)
    if pts.size == 0:
        raise ValueError(f"空点云: {path}")
    return pts.astype(np.float32)


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


def _score_partial(
    partial_pcd: Path,
    complete_pcd: Path,
    *,
    num_views: int,
    hpr_sphere_r: float,
    hpr_r_eff: float,
) -> tuple[int, int]:
    part = _read_pcd_xyz(partial_pcd)
    gt_obj = _read_pcd_xyz(complete_pcd)
    _part_cano, gt_cano, _c, _s = _prep.normalize_by_complete(gt_obj, part)
    return max_hpr_visibility_count(
        gt_cano,
        num_views=num_views,
        hpr_sphere_r=hpr_sphere_r,
        hpr_radius_eff=hpr_r_eff,
    )


def _sanitize_invalid_thread_env() -> None:
    """去掉空串/0/非整数等非法值，避免 libgomp 报 Invalid OMP_NUM_THREADS。"""
    keys = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_NUM_THREADS",
    )
    for k in keys:
        v = os.environ.get(k)
        if v is None:
            continue
        s = str(v).strip()
        if not s:
            del os.environ[k]
            continue
        try:
            n = int(float(s.split(",")[0].strip()))
        except ValueError:
            del os.environ[k]
            continue
        if n < 1:
            del os.environ[k]


def _limit_blas_omp_threads(n: int) -> None:
    s = str(max(1, int(n)))
    for k in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[k] = s


def _pool_init_threads(omp_threads: int) -> None:
    _sanitize_invalid_thread_env()
    _limit_blas_omp_threads(max(1, int(omp_threads)))


def _best_view_for_model(
    split: str,
    synset: str,
    model_id: str,
    candidates: list[tuple[Path, Path, int]],
    pcn_root: Path,
    *,
    num_views: int,
    hpr_sphere_r: float,
    hpr_r_eff: float,
) -> dict | None:
    best_pp: Path | None = None
    best_vi = 10**9
    best_n = -1
    best_dir = 0
    for partial_pcd, complete_pcd, view_idx in candidates:
        try:
            n_vis, dir_i = _score_partial(
                partial_pcd,
                complete_pcd,
                num_views=num_views,
                hpr_sphere_r=hpr_sphere_r,
                hpr_r_eff=hpr_r_eff,
            )
        except Exception as e:
            print(f"\n[warn] {partial_pcd}: {e}", file=sys.stderr)
            continue
        if n_vis > best_n or (n_vis == best_n and int(view_idx) < int(best_vi)):
            best_n = n_vis
            best_vi = int(view_idx)
            best_dir = dir_i
            best_pp = partial_pcd

    if best_pp is None:
        return None

    stem_full = sample_stem(
        split=split, synset=synset, model_id=model_id, view=best_vi
    )
    rel = best_pp.relative_to(pcn_root)
    return {
        "split": split,
        "synset": synset,
        "model_id": model_id,
        "best_view_idx": int(best_vi),
        "best_partial_file": best_pp.name,
        "best_partial_relpath": str(rel).replace("\\", "/"),
        "best_hpr_dir_idx": int(best_dir),
        "n_visible_hpr_max": int(best_n),
        "sample_stem": stem_full,
        "n_partial_candidates": len(candidates),
    }


def _mp_pack_candidates(
    cands: list[tuple[Path, Path, int]],
) -> list[tuple[str, str, int]]:
    return [(str(pp), str(gp), int(vi)) for pp, gp, vi in cands]


def _mp_worker_model(
    packed: tuple[tuple[str, str, str], list[tuple[str, str, int]]],
    cfg: dict[str, object],
) -> tuple[str, dict | None]:
    key, cand_s = packed
    split, synset, model_id = key
    pcn_root = Path(str(cfg["pcn_root"]))
    candidates = [(Path(a), Path(b), int(c)) for a, b, c in cand_s]
    entry = _best_view_for_model(
        split,
        synset,
        model_id,
        candidates,
        pcn_root,
        num_views=int(cfg["num_views"]),
        hpr_sphere_r=float(cfg["hpr_sphere_r"]),
        hpr_r_eff=float(cfg["hpr_r_eff"]),
    )
    jk = f"{split}__{synset}__{model_id}"
    return (jk, entry)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="PCN：按 HPR 最大可见 complete 点数选最优 partial 视角"
    )
    ap.add_argument("--pcn-root", type=str, default=os.path.join(_PROJECT_ROOT, "PCN"))
    ap.add_argument(
        "--out-json",
        type=str,
        default=os.path.join(_PROJECT_ROOT, "data", "processed", "PCN_hpr_best_views.json"),
    )
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--num-views", type=int, default=32)
    ap.add_argument("--hpr-sphere-r", type=float, default=3.0)
    ap.add_argument("--hpr-radius", type=float, default=100.0)
    ap.add_argument("--hpr-radius-factor", type=float, default=25.0)
    ap.add_argument(
        "--pca-axis",
        default="z",
        choices=("x", "y", "z"),
        help="[deprecated] 新管线不做 PCA；保留以兼容旧命令，已忽略",
    )
    ap.add_argument("--max-models", type=int, default=0)
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="并行进程数；0=自动（约 min(CPU 核数, 48)）；1=串行无进程池",
    )
    ap.add_argument(
        "--omp-threads-per-worker",
        type=int,
        default=1,
        help="每个子进程内 OpenMP/BLAS 等线程上限；多进程时建议保持为 1",
    )
    args = ap.parse_args()
    _sanitize_invalid_thread_env()

    pcn_root = Path(args.pcn_root).resolve()
    if not pcn_root.is_dir():
        print(f"错误: --pcn-root 不存在: {pcn_root}", file=sys.stderr)
        return 1

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    pairs = _iter_pairs(pcn_root, splits)
    if not pairs:
        print("无 (partial, complete) 对", file=sys.stderr)
        return 1

    by_model: dict[tuple[str, str, str], list[tuple[Path, Path, int]]] = defaultdict(list)
    for split, synset, model_id, pp, gp, vi in pairs:
        by_model[(split, synset, model_id)].append((pp, gp, vi))

    keys_sorted = sorted(by_model.keys())
    if args.max_models > 0:
        keys_sorted = keys_sorted[: args.max_models]

    hpr_r_eff = hpr_radius_effective(
        args.hpr_sphere_r, args.hpr_radius, args.hpr_radius_factor
    )

    cpu_n = os.cpu_count() or 1
    if int(args.workers) <= 0:
        n_workers = max(1, min(cpu_n, 48))
    else:
        n_workers = max(1, int(args.workers))
    omp_tw = max(1, int(args.omp_threads_per_worker))

    out_by_model: dict[str, dict] = {}
    if n_workers == 1:
        for key in tqdm(keys_sorted, desc="pcn best hpr view"):
            split, synset, model_id = key
            entry = _best_view_for_model(
                split,
                synset,
                model_id,
                by_model[key],
                pcn_root,
                num_views=args.num_views,
                hpr_sphere_r=args.hpr_sphere_r,
                hpr_r_eff=hpr_r_eff,
            )
            if entry is None:
                continue
            jk = f"{split}__{synset}__{model_id}"
            out_by_model[jk] = entry
    else:
        tasks: list[tuple[tuple[str, str, str], list[tuple[str, str, int]]]] = []
        for key in keys_sorted:
            tasks.append((key, _mp_pack_candidates(by_model[key])))
        cfg_dict: dict[str, object] = {
            "pcn_root": str(pcn_root),
            "num_views": int(args.num_views),
            "hpr_sphere_r": float(args.hpr_sphere_r),
            "hpr_r_eff": float(hpr_r_eff),
        }
        n_task = len(tasks)
        worker_fn = partial(_mp_worker_model, cfg=cfg_dict)
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_pool_init_threads,
            initargs=(omp_tw,),
        ) as ex:
            futures = [ex.submit(worker_fn, t) for t in tasks]
            for fut in tqdm(
                as_completed(futures),
                total=n_task,
                desc=f"pcn best hpr view [{n_workers}w]",
            ):
                jk, entry = fut.result()
                if entry is not None:
                    out_by_model[jk] = entry

    payload = {
        "version": 1,
        "description": "Per-model best partial .pcd by max HPR visibility (bbox+PCA on partial, same as legacy preprocess).",
        "params": {
            "pcn_root": str(pcn_root),
            "num_views": args.num_views,
            "hpr_sphere_r": args.hpr_sphere_r,
            "hpr_radius": args.hpr_radius,
            "hpr_radius_factor": args.hpr_radius_factor,
            "hpr_radius_effective": hpr_r_eff,
            "pca_axis": args.pca_axis,
            "workers": n_workers,
            "omp_threads_per_worker": omp_tw,
            "cpu_count": cpu_n,
        },
        "by_model": out_by_model,
    }

    out_path = Path(args.out_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"写入 {len(out_by_model)} 条 -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
