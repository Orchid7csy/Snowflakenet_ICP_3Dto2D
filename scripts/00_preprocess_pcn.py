"""
PCN 预处理（新 schema，与 PCN 预训练分布一致）：
  partial/complete 读入（物体系）→ 在物体系按 complete 的 AABB 中心 + max-radius 单位球
  归一化得到 canonical input/gt（不再做 PCA 重定向）→ 重采样为 input=2048 / gt=16384；
  随机 T_far 仅作用于 obs_w（用于位姿估计任务），meta 同时保存
  canonical (centroid_cano/scale_cano/可选 R_aug) 与世界刚体 (R_far/t_far/T_far_4x4)。

策略：
  - 废弃 HPR 选择，固定使用每个模型的 8 个标准视角（view_idx ∈ [0,7]）。
  - input 点数强制为 2048（扰动后再采样），保证与 SNet 输入契约一致。
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import zlib
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


def _filter_pairs_to_fixed_8_views(
    pairs: list[tuple[str, str, str, Path, Path, int]]
) -> list[tuple[str, str, str, Path, Path, int]]:
    return [item for item in pairs if 0 <= int(item[5]) < 8]


def _process_one_with_gt(
    split: str,
    synset: str,
    model_id: str,
    partial_pcd: Path,
    complete_pcd: Path,
    view_idx: int,
    part_obj: np.ndarray,
    gt_obj: np.ndarray,
    *,
    num_input: int,
    num_gt: int,
    base_seed: int,
    t_min: float,
    t_max: float,
    rot_aug_deg: float,
    rot_aug_axis: str,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, dict]:
    stem = sample_stem(split=split, synset=synset, model_id=model_id, view=view_idx)
    rng = np.random.default_rng(
        (base_seed + zlib.adler32(stem.encode("utf-8"))) % (2**32)
    )

    part_cano, gt_cano, c_cano, scale_cano = prep.normalize_by_complete(gt_obj, part_obj)

    r_aug = np.eye(3, dtype=np.float32)
    if rot_aug_deg > 0.0:
        r_aug = prep.random_gravity_axis_rot(rng, rot_aug_deg, axis=rot_aug_axis)
        part_cano = (part_cano @ r_aug.T).astype(np.float32)
        gt_cano = (gt_cano @ r_aug.T).astype(np.float32)

    r_far, t_far, t_norm = prep.sample_random_far_transform(rng, t_min, t_max)
    obs_w = prep.apply_rigid_row(part_obj, r_far, t_far)

    p_rs = prep.resample_rng(part_cano, num_input, rng)
    g_rs = prep.resample_rng(gt_cano, num_gt, rng)
    obs_rs = prep.resample_rng(obs_w, num_input, rng)
    t_far_4x4 = prep.rigid_T_4x4(r_far, t_far)
    meta = {
        "C_cano": c_cano,
        "centroid_cano": c_cano,
        "scale_cano": np.float32(scale_cano),
        "R_aug": r_aug,
        "rot_aug_deg": np.float32(rot_aug_deg),
        "rot_aug_axis": rot_aug_axis,
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


def _run_one_model(
    model_key: tuple[str, str, str],
    candidates: list[tuple[Path, Path, int]],
    out_root: Path,
    *,
    num_input: int,
    num_gt: int,
    base_seed: int,
    t_min: float,
    t_max: float,
    rot_aug_deg: float,
    rot_aug_axis: str,
    skip_existing: bool,
    no_meta: bool,
) -> tuple[int, int, int, list[str]]:
    """处理同一 model 下的所有候选 partial：仅读 complete 一次。

    返回 (n_ok, n_skip, n_fail, fail_msgs)。
    """
    if not candidates:
        return (0, 0, 0, [])
    split, synset, model_id = model_key
    fail_msgs: list[str] = []

    idir = out_root / split / "input"
    gdir = out_root / split / "gt"
    odir = out_root / split / "obs_w"
    mdir = out_root / split / "meta"
    for d in (idir, gdir, odir):
        d.mkdir(parents=True, exist_ok=True)
    if not no_meta:
        mdir.mkdir(parents=True, exist_ok=True)

    todo: list[tuple[Path, Path, int, str, Path, Path, Path, Path]] = []
    n_skip = 0
    for pp, gp, view_idx in candidates:
        stem = sample_stem(
            split=split, synset=synset, model_id=model_id, view=view_idx
        )
        ip = idir / f"{stem}.npy"
        gpout = gdir / f"{stem}.npy"
        op = odir / f"{stem}.npy"
        mp = mdir / f"{stem}.npz"
        if skip_existing and ip.is_file() and gpout.is_file() and op.is_file():
            n_skip += 1
            continue
        todo.append((pp, gp, int(view_idx), stem, ip, gpout, op, mp))

    if not todo:
        return (0, n_skip, 0, fail_msgs)

    complete_pcd = todo[0][1]
    complete_resolved = complete_pcd.resolve()
    for pp, gp, *_ in todo:
        if gp.resolve() != complete_resolved:
            fail_msgs.append(
                f"{split}/{synset}/{model_id}: complete 路径不一致: {gp} vs {complete_pcd}"
            )
            return (0, n_skip, len(todo), fail_msgs)

    try:
        gt_obj = read_pcd_xyz(complete_pcd)
    except Exception as e:
        fail_msgs.append(f"{complete_pcd}: {e}")
        return (0, n_skip, len(todo), fail_msgs)

    n_ok = 0
    n_fail = 0
    for pp, gp, view_idx, stem, ip, gpout, op, mp in todo:
        try:
            part_obj = read_pcd_xyz(pp)
        except Exception as e:
            n_fail += 1
            fail_msgs.append(f"{pp}: {e}")
            continue
        try:
            _stem, pin, gto, obw, meta = _process_one_with_gt(
                split,
                synset,
                model_id,
                pp,
                gp,
                view_idx,
                part_obj,
                gt_obj,
                num_input=num_input,
                num_gt=num_gt,
                base_seed=base_seed,
                t_min=t_min,
                t_max=t_max,
                rot_aug_deg=rot_aug_deg,
                rot_aug_axis=rot_aug_axis,
            )
        except Exception as e:
            n_fail += 1
            fail_msgs.append(f"{pp}: {e}")
            continue
        if _stem != stem:
            n_fail += 1
            fail_msgs.append(f"internal stem mismatch: {_stem!r} vs {stem!r}")
            continue
        try:
            np.save(str(ip), pin.astype(np.float32))
            np.save(str(gpout), gto.astype(np.float32))
            np.save(str(op), obw.astype(np.float32))
            if not no_meta:
                np.savez(str(mp), **meta)
            n_ok += 1
        except Exception as e:
            n_fail += 1
            fail_msgs.append(f"save {stem}: {e}")
    return (n_ok, n_skip, n_fail, fail_msgs)


def _mp_pack_model(
    key: tuple[str, str, str],
    cands: list[tuple[Path, Path, int]],
) -> tuple[tuple[str, str, str], list[tuple[str, str, int]]]:
    return (key, [(str(pp), str(gp), int(vi)) for pp, gp, vi in cands])


def _mp_worker_model(
    packed: tuple[tuple[str, str, str], list[tuple[str, str, int]]],
    cfg: dict[str, object],
) -> tuple[int, int, int, list[str]]:
    key, cand_s = packed
    candidates = [(Path(a), Path(b), int(c)) for a, b, c in cand_s]
    return _run_one_model(
        key,
        candidates,
        Path(str(cfg["out_root"])),
        num_input=int(cfg["num_input"]),
        num_gt=int(cfg["num_gt"]),
        base_seed=int(cfg["base_seed"]),
        t_min=float(cfg["t_min"]),
        t_max=float(cfg["t_max"]),
        rot_aug_deg=float(cfg["rot_aug_deg"]),
        rot_aug_axis=str(cfg["rot_aug_axis"]),
        skip_existing=bool(cfg["skip_existing"]),
        no_meta=bool(cfg["no_meta"]),
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="PCN：按 complete 归一化的 canonical input/gt + T_far→obs_w + meta"
    )
    ap.add_argument("--pcn-root", type=str, default=os.path.join(_PROJECT_ROOT, "PCN"))
    ap.add_argument(
        "--out-root",
        type=str,
        default="",
        help="默认 hard: data/processed/PCN_far_cano_in{input}_gt{gt}/；"
             "easy: data/processed/PCN_hpr_easy_cano_in{input}_gt{gt}/",
    )
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--num-input", type=int, default=2048)
    ap.add_argument("--num-gt", type=int, default=16384)
    ap.add_argument(
        "--pca-axis",
        default="z",
        choices=("x", "y", "z"),
        help="[deprecated] 新管线不做 PCA，仅占位兼容旧命令；忽略",
    )
    ap.add_argument(
        "--rot-aug-deg",
        type=float,
        default=0.0,
        help="canonical 系内绕重力轴的随机小角度（度）；0 表示关闭",
    )
    ap.add_argument(
        "--rot-aug-axis",
        default="z",
        choices=("x", "y", "z"),
        help="--rot-aug-deg 对应的重力轴",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t-min", type=float, default=1.0, help="t_far 模长下界")
    ap.add_argument("--t-max", type=float, default=3.0, help="t_far 模长上界")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--max-samples", type=int, default=0)
    ap.add_argument("--no-meta", action="store_true")
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
    if int(args.num_input) != 2048:
        raise ValueError("SNet 输入点数必须固定为 2048，请使用 --num-input 2048。")

    pcn_root = Path(args.pcn_root).resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not args.out_root:
        name = f"PCN_far8_cano_in{args.num_input}_gt{args.num_gt}"
        out_root = Path(_PROJECT_ROOT, "data", "processed", name)
    else:
        out_root = Path(args.out_root).resolve()

    pairs = _iter_pairs(pcn_root, splits)
    n_pairs_before_selection = len(pairs)
    pairs = _filter_pairs_to_fixed_8_views(pairs)
    if args.max_samples > 0:
        pairs = pairs[: args.max_samples]

    if not pairs:
        print("无 (partial, complete) 对", file=sys.stderr)
        return 1

    by_model: dict[tuple[str, str, str], list[tuple[Path, Path, int]]] = defaultdict(list)
    for split, synset, model_id, pp, gp, vi in pairs:
        by_model[(split, synset, model_id)].append((pp, gp, vi))
    model_keys = sorted(by_model.keys())
    n_models = len(model_keys)

    cpu_n = os.cpu_count() or 1
    if int(args.workers) <= 0:
        n_workers = max(1, min(cpu_n, 48))
    else:
        n_workers = max(1, int(args.workers))
    omp_tw = max(1, int(args.omp_threads_per_worker))

    n_ok, n_skip, n_fail = 0, 0, 0
    desc = "pcn fixed-8-views cano+T_far"
    if n_workers == 1:
        for key in tqdm(model_keys, desc=desc):
            ok, sk, fa, fmsgs = _run_one_model(
                key,
                by_model[key],
                out_root,
                num_input=args.num_input,
                num_gt=args.num_gt,
                base_seed=args.seed,
                t_min=args.t_min,
                t_max=args.t_max,
                rot_aug_deg=float(args.rot_aug_deg),
                rot_aug_axis=str(args.rot_aug_axis),
                skip_existing=bool(args.skip_existing),
                no_meta=bool(args.no_meta),
            )
            n_ok += ok
            n_skip += sk
            n_fail += fa
            for m in fmsgs:
                print(f"\n[fail] {m}", file=sys.stderr)
    else:
        cfg_dict: dict[str, object] = {
            "out_root": str(out_root),
            "num_input": int(args.num_input),
            "num_gt": int(args.num_gt),
            "base_seed": int(args.seed),
            "t_min": float(args.t_min),
            "t_max": float(args.t_max),
            "rot_aug_deg": float(args.rot_aug_deg),
            "rot_aug_axis": str(args.rot_aug_axis),
            "skip_existing": bool(args.skip_existing),
            "no_meta": bool(args.no_meta),
        }
        mp_items = [_mp_pack_model(k, by_model[k]) for k in model_keys]
        worker_fn = partial(_mp_worker_model, cfg=cfg_dict)
        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_pool_init_threads,
            initargs=(omp_tw,),
        ) as ex:
            futures = [ex.submit(worker_fn, it) for it in mp_items]
            for fut in tqdm(
                as_completed(futures),
                total=n_models,
                desc=f"{desc} [{n_workers}w]",
            ):
                ok, sk, fa, fmsgs = fut.result()
                n_ok += ok
                n_skip += sk
                n_fail += fa
                for m in fmsgs:
                    print(f"\n[fail] {m}", file=sys.stderr)

    manifest = out_root / "00_preprocess_pcn_manifest.txt"
    with open(manifest, "w", encoding="utf-8") as f:
        f.write(f"pcn_root={pcn_root}\nout_root={out_root}\n")
        f.write(
            f"ok={n_ok} skip={n_skip} fail={n_fail} pairs={len(pairs)} "
            f"input={args.num_input} gt={args.num_gt} t_min={args.t_min} t_max={args.t_max}\n"
        )
        f.write(
            f"view_policy=fixed_8_views(view_idx in [0,7]) "
            f"pairs_before_selection={n_pairs_before_selection}\n"
        )
        f.write(
            f"schema=cano rot_aug_deg={args.rot_aug_deg} rot_aug_axis={args.rot_aug_axis}\n"
        )
        f.write(
            f"workers={n_workers} omp_threads_per_worker={omp_tw} cpu_count={cpu_n} "
            f"n_models={n_models}\n"
        )
    print(f"完成: 写入 {n_ok}, 跳过 {n_skip}, 失败 {n_fail} / {len(pairs)}. 输出: {out_root}\n摘要: {manifest}")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
