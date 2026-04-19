"""
scripts/stat_itodd_preprocess.py

对 ITODD completion 预处理结果做量化统计（不写回数据）：

1) input–gt Chamfer-L1：与 scripts/03_train_completion_itodd.py 中训练损失一致，
   可视为「无网络时几何上 input 与完整形状差多少」的参考上界。
2) GT 往返：saved_gt → 逆 PCA + 反 bbox → 再正向变换，应与 saved_gt 重合（数值误差）。
3) obs–input 逆变换一致性：input 经逆变换回到 obs 坐标系，与 obs 的 Chamfer（检测流水线是否自洽）。
4) meta 分布：scale、missing_rate、n_visible 等摘要。

用法（--root 必须是本机真实路径，不要用文档里的占位符 /path/to/...）:
  python scripts/stat_itodd_preprocess.py --root data/processed/itodd --split both
  python scripts/stat_itodd_preprocess.py --root data/processed/itodd --split test --max-samples 500

支持两种目录结构:
  A) <root>/train/input 与 <root>/test/input（与 00_preprocess_itodd.py 一致）
  B) <root>/input（仅一个 split 时）

定位 / 导出:
  --export-dir DIR   写入 <split>_per_sample.csv 与 <split>_outliers.csv（便于筛脏数据、对照 view_npy）
  --thr-obs 等       异常阈值（见 --help）
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
snet_root = os.path.join(project_root, "Snet", "SnowflakeNet-main")
for p in (project_root, snet_root):
    if p not in sys.path:
        sys.path.insert(0, p)

import torch
from loss_functions.Chamfer3D.dist_chamfer_3D import chamfer_3DDist

_chamfer = chamfer_3DDist()


def _npz_has(z, key: str) -> bool:
    return key in getattr(z, "files", ())


def _meta_float(z, *keys: str) -> Optional[float]:
    for k in keys:
        if _npz_has(z, k):
            return float(np.asarray(z[k]).reshape(()))
    return None


def infer_scale_from_obs_input(
    obs: np.ndarray,
    inp: np.ndarray,
    C_bbox: np.ndarray,
    R_pca: np.ndarray,
    mu_pca: np.ndarray,
) -> float:
    """
    当 meta 中无 scale 时：由 p_obs = p_norm*scale + C 与 p_norm = inv_pca(input) 估计标量 scale。
    """
    p_norm = invert_pca_rigid_np(inp, R_pca, mu_pca)
    diff = obs.astype(np.float64) - C_bbox.reshape(1, 3)
    pn = np.linalg.norm(p_norm, axis=1)
    dn = np.linalg.norm(diff, axis=1)
    ratios = dn / (pn + 1e-8)
    return float(np.median(ratios))


def load_bbox_scale_pca(z, obs: np.ndarray, inp: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, bool]:
    C_bbox = np.asarray(z["C_bbox"], dtype=np.float32).reshape(3)
    R_pca = np.asarray(z["R_pca"], dtype=np.float32).reshape(3, 3)
    mu_pca = np.asarray(z["mu_pca"], dtype=np.float32).reshape(3)
    scale = _meta_float(z, "scale", "bbox_scale", "norm_scale")
    from_meta = scale is not None
    if scale is None:
        scale = infer_scale_from_obs_input(obs, inp, C_bbox, R_pca, mu_pca)
    return C_bbox, float(scale), R_pca, mu_pca, from_meta


def invert_pca_rigid_np(aligned: np.ndarray, R: np.ndarray, mu: np.ndarray) -> np.ndarray:
    pts = np.asarray(aligned, dtype=np.float32)
    Rm = np.asarray(R, dtype=np.float32).reshape(3, 3)
    muv = np.asarray(mu, dtype=np.float32).reshape(1, 3)
    return ((pts - muv) @ Rm + muv).astype(np.float32)


def apply_pca_rigid_np(points: np.ndarray, R: np.ndarray, mu: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    R64 = np.asarray(R, dtype=np.float64).reshape(3, 3)
    mu64 = np.asarray(mu, dtype=np.float64).reshape(1, 3)
    return ((R64 @ (pts - mu64).T).T + mu64).astype(np.float32)


def discover_split_bases(root: str) -> List[Tuple[str, str]]:
    """
    返回 [(标签, 该 split 的根目录)], 目录下需含 input/gt/obs/meta。
    """
    out: List[Tuple[str, str]] = []
    train_in = os.path.join(root, "train", "input")
    if os.path.isdir(train_in):
        for name in ("train", "test"):
            p = os.path.join(root, name)
            if os.path.isdir(os.path.join(p, "input")):
                out.append((name, p))
        return out
    if os.path.isdir(os.path.join(root, "input")):
        out.append(("all", root))
    return out


def list_stems(split_base: str) -> List[str]:
    d = os.path.join(split_base, "input")
    if not os.path.isdir(d):
        raise FileNotFoundError(d)
    return sorted(os.path.splitext(f)[0] for f in os.listdir(d) if f.endswith(".npy"))


def summarize(name: str, arr: np.ndarray) -> None:
    a = np.asarray(arr, dtype=np.float64)
    print(f"\n=== {name} (n={a.size}) ===")
    if a.size == 0:
        return
    valid = a[np.isfinite(a)]
    if valid.size == 0:
        print("  (no finite values)")
        return
    for q in (0, 5, 25, 50, 75, 95, 100):
        print(f"  p{q:02d}: {np.percentile(valid, q):.6f}")
    print(f"  mean: {valid.mean():.6f}  std: {valid.std():.6f}  min: {valid.min():.6f}  max: {valid.max():.6f}")


def report_outliers(
    split_label: str,
    stats: Dict[str, Any],
    thr_obs: float,
    thr_cham: float,
    thr_gt_rt: float,
    top_k: int,
) -> None:
    stem_keys = stats["stem_keys"]
    obs = stats["cd_obs_recon"]
    cham = stats["cd_inp_gt"]
    rt = stats["cd_roundtrip_gt"]
    has_meta = stats["has_scale_meta"]
    bad_obs = obs > thr_obs
    bad_cham = cham > thr_cham
    bad_rt = rt > thr_gt_rt
    any_bad = bad_obs | bad_cham | bad_rt
    n = len(stem_keys)
    print(f"\n--- 定位 [{split_label}] 阈值: max_err_obs>{thr_obs:g} | chamfer_l1>{thr_cham:g} | gt_roundtrip>{thr_gt_rt:g} ---")
    print(f"  max_err_obs 超阈值: {int(bad_obs.sum())}")
    print(f"  chamfer_l1 超阈值: {int(bad_cham.sum())}")
    print(f"  gt_roundtrip 超阈值: {int(bad_rt.sum())}")
    print(f"  任一超阈值: {int(any_bad.sum())} / {n}")

    idx_missing = ~has_meta
    if np.any(idx_missing):
        bad_o_miss = np.logical_and(bad_obs, idx_missing)
        print(f"  [npz 无 scale、曾用 obs 反推] 共 {int(idx_missing.sum())} 条，其中 obs 超阈值 {int(bad_o_miss.sum())} 条")

    k = min(top_k, n)

    def print_top(title: str, metric: np.ndarray) -> None:
        order = np.argsort(-metric)[:k]
        print(f"  {title} (top {k}):")
        for i in order:
            hm = "Y" if has_meta[i] else "N"
            print(f"    {stem_keys[i]}\t{metric[i]:.6f}\tscale_in_meta={hm}")

    print_top("chamfer_l1(input,gt) 最大", cham)
    print_top("max_err_obs_recon 最大", obs)


def _ensure_parent_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def write_per_sample_csv(path: str, split_label: str, stats: Dict[str, Any]) -> None:
    _ensure_parent_dir(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "split",
                "stem",
                "chamfer_l1_input_gt",
                "max_err_gt_roundtrip",
                "max_err_obs_recon",
                "frob_R_minus_I",
                "bbox_scale",
                "has_scale_in_meta",
                "missing_rate",
                "n_visible",
            ]
        )
        n = len(stats["stem_keys"])
        for i in range(n):
            w.writerow(
                [
                    split_label,
                    stats["stem_keys"][i],
                    f"{stats['cd_inp_gt'][i]:.8f}",
                    f"{stats['cd_roundtrip_gt'][i]:.8f}",
                    f"{stats['cd_obs_recon'][i]:.8f}",
                    f"{stats['frob_r_eye'][i]:.8f}",
                    f"{stats['scale'][i]:.8f}",
                    int(stats["has_scale_meta"][i]),
                    f"{stats['missing_rate'][i]:.8f}" if np.isfinite(stats["missing_rate"][i]) else "",
                    f"{stats['n_visible'][i]:.8f}" if np.isfinite(stats["n_visible"][i]) else "",
                ]
            )


def flag_rows(
    stats: Dict[str, Any], thr_obs: float, thr_cham: float, thr_gt_rt: float
) -> np.ndarray:
    return (
        (stats["cd_obs_recon"] > thr_obs)
        | (stats["cd_inp_gt"] > thr_cham)
        | (stats["cd_roundtrip_gt"] > thr_gt_rt)
    )


def write_outliers_csv(
    path: str,
    split_label: str,
    stats: Dict[str, Any],
    thr_obs: float,
    thr_cham: float,
    thr_gt_rt: float,
) -> None:
    mask = flag_rows(stats, thr_obs, thr_cham, thr_gt_rt)
    _ensure_parent_dir(path)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "split",
                "stem",
                "chamfer_l1_input_gt",
                "max_err_gt_roundtrip",
                "max_err_obs_recon",
                "bbox_scale",
                "has_scale_in_meta",
                "reason",
            ]
        )
        n = len(stats["stem_keys"])
        for i in range(n):
            if not mask[i]:
                continue
            reasons = []
            if stats["cd_obs_recon"][i] > thr_obs:
                reasons.append("obs_recon")
            if stats["cd_inp_gt"][i] > thr_cham:
                reasons.append("chamfer")
            if stats["cd_roundtrip_gt"][i] > thr_gt_rt:
                reasons.append("gt_roundtrip")
            w.writerow(
                [
                    split_label,
                    stats["stem_keys"][i],
                    f"{stats['cd_inp_gt'][i]:.8f}",
                    f"{stats['cd_roundtrip_gt'][i]:.8f}",
                    f"{stats['cd_obs_recon'][i]:.8f}",
                    f"{stats['scale'][i]:.8f}",
                    int(stats["has_scale_meta"][i]),
                    "+".join(reasons),
                ]
            )


def collect_for_split(
    split_base: str,
    max_samples: Optional[int],
    batch_size: int,
    device: torch.device,
) -> Dict[str, Any]:
    stems = list_stems(split_base)
    if max_samples is not None:
        stems = stems[: max(0, max_samples)]

    stem_keys: List[str] = []
    cd_inp_gt: List[float] = []
    cd_roundtrip_gt: List[float] = []
    cd_obs_recon: List[float] = []
    frob_r_eye: List[float] = []
    scales: List[float] = []
    missing_rates: List[float] = []
    n_visible: List[float] = []
    has_scale_meta: List[bool] = []
    n_scale_inferred = 0

    n_batches = (len(stems) + batch_size - 1) // batch_size
    for bi in range(n_batches):
        chunk = stems[bi * batch_size : (bi + 1) * batch_size]
        if not chunk:
            break

        inputs: List[np.ndarray] = []
        gts: List[np.ndarray] = []

        for stem in chunk:
            stem_keys.append(stem)
            inp = np.load(os.path.join(split_base, "input", f"{stem}.npy"))
            gt = np.load(os.path.join(split_base, "gt", f"{stem}.npy"))
            obs = np.load(os.path.join(split_base, "obs", f"{stem}.npy"))
            meta = np.load(os.path.join(split_base, "meta", f"{stem}.npz"), allow_pickle=True)

            C_bbox, scale, R_pca, mu_pca, scale_from_meta = load_bbox_scale_pca(meta, obs, inp)
            has_scale_meta.append(scale_from_meta)
            if not scale_from_meta:
                n_scale_inferred += 1

            inputs.append(inp.astype(np.float32))
            gts.append(gt.astype(np.float32))

            gt_norm = invert_pca_rigid_np(gt, R_pca, mu_pca)
            gt_w_hat = gt_norm * scale + C_bbox[None, :]
            gt_norm_back = (gt_w_hat - C_bbox[None, :]) / scale
            gt_hat = apply_pca_rigid_np(gt_norm_back, R_pca, mu_pca)
            err = np.linalg.norm(gt_hat - gt, axis=1).max()
            cd_roundtrip_gt.append(float(err))

            inp_sync = invert_pca_rigid_np(inp, R_pca, mu_pca)
            obs_hat = inp_sync * scale + C_bbox[None, :]
            err_o = np.linalg.norm(obs_hat - obs, axis=1).max()
            cd_obs_recon.append(float(err_o))

            r = R_pca.astype(np.float64)
            frob_r_eye.append(float(np.linalg.norm(r - np.eye(3))))

            scales.append(scale)
            if _npz_has(meta, "missing_rate"):
                missing_rates.append(float(np.asarray(meta["missing_rate"]).reshape(())))
            else:
                missing_rates.append(float("nan"))
            if _npz_has(meta, "n_visible"):
                n_visible.append(float(np.asarray(meta["n_visible"]).reshape(())))
            else:
                n_visible.append(float("nan"))

        tin = torch.from_numpy(np.stack(inputs, axis=0)).to(device)
        tgt = torch.from_numpy(np.stack(gts, axis=0)).to(device)
        with torch.no_grad():
            dist1, dist2, _, _ = _chamfer(tin, tgt)
            eps = 1e-8
            per = (dist1 + eps).sqrt().mean(dim=1) + (dist2 + eps).sqrt().mean(dim=1)
            cd_inp_gt.extend(per.cpu().numpy().tolist())

    return {
        "stems": len(stems),
        "stem_keys": stem_keys,
        "cd_inp_gt": np.array(cd_inp_gt, dtype=np.float64),
        "cd_roundtrip_gt": np.array(cd_roundtrip_gt, dtype=np.float64),
        "cd_obs_recon": np.array(cd_obs_recon, dtype=np.float64),
        "frob_r_eye": np.array(frob_r_eye, dtype=np.float64),
        "scale": np.array(scales, dtype=np.float64),
        "missing_rate": np.array(missing_rates, dtype=np.float64),
        "n_visible": np.array(n_visible, dtype=np.float64),
        "has_scale_meta": np.asarray(has_scale_meta, dtype=bool),
        "n_scale_inferred": n_scale_inferred,
    }


def main():
    ap = argparse.ArgumentParser(description="ITODD 预处理量化统计")
    ap.add_argument(
        "--root",
        type=str,
        required=True,
        help="预处理根目录：需含 train/input 与 test/input，或仅含 input/（不要用文档占位路径）",
    )
    ap.add_argument("--split", type=str, default="both", choices=("train", "test", "both"))
    ap.add_argument("--max-samples", type=int, default=None, help="每个 split 最多统计多少条（默认全部）")
    ap.add_argument("--batch-size", type=int, default=32, help="Chamfer 批大小（受显存限制）")
    ap.add_argument("--cpu", action="store_true", help="强制 CPU（慢）")
    ap.add_argument(
        "--thr-obs",
        type=float,
        default=1e-3,
        help="max_err_obs_recon 超过则计为异常（默认 1e-3，与 float32 噪声量级）",
    )
    ap.add_argument(
        "--thr-chamfer",
        type=float,
        default=2.0,
        help="chamfer_l1(input,gt) 超过则计为异常（默认 2.0，可按 p99 调）",
    )
    ap.add_argument(
        "--thr-gt-roundtrip",
        type=float,
        default=1e-4,
        help="gt 往返误差超过则计为异常（默认 1e-4）",
    )
    ap.add_argument("--top-k", type=int, default=5, help="打印各指标最高的 stem 条数")
    ap.add_argument(
        "--export-dir",
        type=str,
        default=None,
        help="若指定，则写入 {split}_per_sample.csv 与 {split}_outliers.csv",
    )
    ap.add_argument("--no-locate", action="store_true", help="不打印定位块、不写 outliers CSV（仍可按 --export-dir 写全量 per_sample）")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    if not os.path.isdir(root):
        print(
            f"错误: 路径不存在或不是目录:\n  {root}\n\n"
            "`--root` 需换成本机上的真实预处理目录（文档里示例 /path/to/your/itodd_processed 仅为占位符）。\n"
            f"例如项目默认: {os.path.join(project_root, 'data', 'processed', 'itodd')}",
            file=sys.stderr,
        )
        sys.exit(1)

    bases = discover_split_bases(root)
    if not bases:
        try:
            listing = os.listdir(root)[:20]
        except OSError:
            listing = []
        print(
            f"错误: 在 {root} 下未找到预期子目录。\n"
            "需要下列之一:\n"
            "  A) <root>/train/input/*.npy 与 <root>/test/input/*.npy\n"
            "  B) <root>/input/*.npy\n"
            f"当前目录内容（前 20 项）: {listing}",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.split == "train":
        bases = [b for b in bases if b[0] == "train"]
    elif args.split == "test":
        bases = [b for b in bases if b[0] == "test"]
    if not bases:
        print(
            "错误: 当前 --split 与目录结构不匹配。\n"
            "若只有单目录 <root>/input/，请使用 --split both。\n"
            "若只有 train 或只有 test，请把 --root 指到含 train/test 的上一级。",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"root={root}")
    print(f"device={device}")
    print(f"将统计 split: {', '.join(f'{n} -> {p}' for n, p in bases)}")
    print(
        "指标说明:\n"
        "  cd_inp_gt     : 与训练相同的 Chamfer-L1(input, gt)，按样本；越小说明观测与目标在同一规范系下越近。\n"
        "  cd_roundtrip_gt : max || roundtrip(gt_saved) - gt_saved ||_2 per point（应接近 0）。\n"
        "  cd_obs_recon  : max || 由 input 逆变换重建的 obs - obs ||_2 per point（应接近 0）。\n"
        "  frob_r_eye    : ||R_pca - I||_F；接近 0 表示该样本几乎未做 PCA 旋转。\n"
    )

    for sp, split_base in bases:
        stats = collect_for_split(split_base, args.max_samples, args.batch_size, device)
        print(f"\n########## split={sp}  samples={stats['stems']} ##########")
        if stats.get("n_scale_inferred", 0) > 0:
            print(
                f"  [meta] {stats['n_scale_inferred']}/{stats['stems']} 个样本的 npz 中无 scale，"
                "已用 obs+input 反推标量 scale（与 00_preprocess_itodd 中 bbox 归一化一致）。"
            )
        summarize("Chamfer-L1(input, gt)  [与 val 同量纲]", stats["cd_inp_gt"])
        summarize("max point err GT roundtrip (should ~0)", stats["cd_roundtrip_gt"])
        summarize("max point err obs <- input inverse (should ~0)", stats["cd_obs_recon"])
        summarize("||R_pca - I||_F", stats["frob_r_eye"])
        summarize("bbox scale", stats["scale"])
        if stats["missing_rate"].size:
            summarize("missing_rate (meta)", stats["missing_rate"])
        if stats["n_visible"].size:
            summarize("n_visible after HPR (meta)", stats["n_visible"])

        if not args.no_locate:
            report_outliers(
                sp,
                stats,
                args.thr_obs,
                args.thr_chamfer,
                args.thr_gt_roundtrip,
                args.top_k,
            )

        if args.export_dir:
            ed = os.path.abspath(args.export_dir)
            os.makedirs(ed, exist_ok=True)
            full_path = os.path.join(ed, f"{sp}_per_sample.csv")
            write_per_sample_csv(full_path, sp, stats)
            print(f"\n  [export] 全量: {full_path}")
            if not args.no_locate:
                out_path = os.path.join(ed, f"{sp}_outliers.csv")
                write_outliers_csv(
                    out_path, sp, stats, args.thr_obs, args.thr_chamfer, args.thr_gt_roundtrip
                )
                n_out = int(flag_rows(stats, args.thr_obs, args.thr_chamfer, args.thr_gt_roundtrip).sum())
                print(f"  [export] 异常: {out_path}  ({n_out} 行)")


if __name__ == "__main__":
    main()
