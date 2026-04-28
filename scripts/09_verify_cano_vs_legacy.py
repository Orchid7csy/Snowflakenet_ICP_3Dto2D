"""
快速验证脚本：在少量 PCN 样本上对比 **旧 bbox+PCA 预处理** 与 **新 canonical 预处理**
下，PCN 预训练 SnowflakeNet 补全的 ``max_norm`` 与 ``CD-L1``，确认坍缩问题已解决。

逻辑参考 ``scripts/08_viz_completion_pretrain_vs_finetune.py`` 的 ``--pretrain-ablation``：
仅加载预训练 ckpt，跑同一 partial 在两种归一化下的补全。

用法::

  PYTHONPATH=. python3 scripts/09_verify_cano_vs_legacy.py \\
      --pcn-root PCN --num-samples 5

  PYTHONPATH=. python3 scripts/09_verify_cano_vs_legacy.py \\
      --pcn-root PCN --num-samples 3 --viz

打印的关键指标：
  - ``pred.max_norm``：预训练补全在 canonical 系内的最大半径，应接近 1.0；旧路径常 << 1（坍缩）。
  - ``CD-L1``：补全与 GT 的对称 chamfer，越小越好。
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_SNET_ROOT = os.path.join(_PROJECT_ROOT, "Snet", "SnowflakeNet-main")
for _p in (_PROJECT_ROOT, _SNET_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.data import preprocessing as prep  # noqa: E402
from src.evaluation.npy_forward import forward_from_npy  # noqa: E402
from src.models.chamfer import chamfer_l1_symmetric  # noqa: E402
from src.models.snet_loader import load_snowflakenet  # noqa: E402
from src.utils.io import read_pcd_xyz  # noqa: E402


def _pick_samples(pcn_root: Path, split: str, k: int) -> list[tuple[Path, Path]]:
    out: list[tuple[Path, Path]] = []
    comp_root = pcn_root / split / "complete"
    if not comp_root.is_dir():
        raise FileNotFoundError(comp_root)
    for tax_dir in sorted(comp_root.iterdir()):
        if not tax_dir.is_dir():
            continue
        for gtp in sorted(tax_dir.glob("*.pcd")):
            pdir = pcn_root / split / "partial" / tax_dir.name / gtp.stem
            if not pdir.is_dir():
                continue
            views = sorted(pdir.glob("*.pcd"))
            if not views:
                continue
            out.append((views[0], gtp))
            if len(out) >= k:
                return out
    return out


def _legacy_preprocess(
    part_obj: np.ndarray, gt_obj: np.ndarray, *, pca_axis: str = "z"
) -> tuple[np.ndarray, np.ndarray, dict]:
    """旧路径：partial AABB 中心 + 单位球 + PCA。input/gt 共用 (c, scale, R_pca)。"""
    p_norm, c_bbox, scale = prep.normalize_by_bbox(part_obj)
    p_in, r_pca, mu_pca = prep.pca_align(p_norm, target_axis=pca_axis)
    gt_norm = ((gt_obj - c_bbox[None, :]) / np.float32(scale)).astype(np.float32)
    gt_out = prep.apply_pca_rigid(gt_norm, r_pca, mu_pca)
    info = {"c": c_bbox, "scale": float(scale), "R_pca": r_pca}
    return p_in.astype(np.float32), gt_out.astype(np.float32), info


def _cano_preprocess(
    part_obj: np.ndarray, gt_obj: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict]:
    """新路径：按 complete 的 AABB 中心 + max-radius 单位球归一化，无 PCA。"""
    part_cano, gt_cano, c, scale = prep.normalize_by_complete(gt_obj, part_obj)
    info = {"c": c, "scale": float(scale)}
    return part_cano, gt_cano, info


def _resample(pts: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if pts.shape[0] >= n:
        idx = rng.choice(pts.shape[0], n, replace=False)
    else:
        idx = rng.choice(pts.shape[0], n, replace=True)
    return pts[idx].astype(np.float32)


def _shift_for_viz(pts: np.ndarray, dx: float) -> np.ndarray:
    return (pts.astype(np.float32) + np.array([dx, 0.0, 0.0], dtype=np.float32)).astype(
        np.float32
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="对比旧 bbox+PCA 与新 cano 预处理下 pretrain SnowflakeNet 的补全表现"
    )
    ap.add_argument("--pcn-root", type=str, default=os.path.join(_PROJECT_ROOT, "PCN"))
    ap.add_argument("--split", default="val", choices=("train", "val", "test"))
    ap.add_argument("--num-samples", type=int, default=5)
    ap.add_argument("--num-input", type=int, default=2048)
    ap.add_argument("--num-gt", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--ckpt-pretrain",
        default=os.path.join(
            _SNET_ROOT, "completion", "checkpoints", "ckpt-best-pcn-cd_l1.pth"
        ),
    )
    ap.add_argument(
        "--device", default="auto", choices=("auto", "cuda", "cpu")
    )
    ap.add_argument(
        "--viz",
        action="store_true",
        help="启用 Open3D 单窗并排可视化（每个样本一个窗口；关闭窗口继续下一样本）",
    )
    args = ap.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    pcn_root = Path(args.pcn_root).expanduser().resolve()
    if not pcn_root.is_dir():
        print(f"错误：--pcn-root 不存在: {pcn_root}", file=sys.stderr)
        return 1

    samples = _pick_samples(pcn_root, args.split, max(1, int(args.num_samples)))
    if not samples:
        print(f"错误：未在 {pcn_root}/{args.split} 找到 (partial, complete) 对", file=sys.stderr)
        return 1

    print(f"device={device} ckpt={args.ckpt_pretrain}")
    print(f"将对比 {len(samples)} 个样本（split={args.split}）\n")
    model = load_snowflakenet(os.path.abspath(args.ckpt_pretrain))
    fwd_kw = dict(
        input_mode="direct", do_comp_filter=False, n_input_points=args.num_input
    )

    rng = np.random.default_rng(int(args.seed))

    legacy_norms: list[float] = []
    legacy_cds: list[float] = []
    cano_norms: list[float] = []
    cano_cds: list[float] = []

    header = (
        f"{'sample':<70} | "
        f"{'legacy.pred_max':>15}  {'legacy.CD-L1':>13} | "
        f"{'cano.pred_max':>13}  {'cano.CD-L1':>11}"
    )
    print(header)
    print("-" * len(header))

    for partial_pcd, complete_pcd in samples:
        tag = f"{partial_pcd.parent.parent.name}/{partial_pcd.parent.name}/{partial_pcd.stem}"
        part_obj = read_pcd_xyz(partial_pcd)
        gt_obj = read_pcd_xyz(complete_pcd)

        p_legacy, g_legacy, _info_l = _legacy_preprocess(part_obj, gt_obj)
        p_cano, g_cano, _info_c = _cano_preprocess(part_obj, gt_obj)

        p_legacy = _resample(p_legacy, args.num_input, rng)
        g_legacy = _resample(g_legacy, args.num_gt, rng)
        p_cano = _resample(p_cano, args.num_input, rng)
        g_cano = _resample(g_cano, args.num_gt, rng)

        pred_legacy = forward_from_npy(model, p_legacy, device, **fwd_kw)
        pred_cano = forward_from_npy(model, p_cano, device, **fwd_kw)

        n_legacy = float(np.linalg.norm(pred_legacy, axis=1).max())
        n_cano = float(np.linalg.norm(pred_cano, axis=1).max())
        cd_legacy = chamfer_l1_symmetric(
            torch.from_numpy(pred_legacy).float().unsqueeze(0).to(device),
            torch.from_numpy(g_legacy).float().unsqueeze(0).to(device),
        ).item()
        cd_cano = chamfer_l1_symmetric(
            torch.from_numpy(pred_cano).float().unsqueeze(0).to(device),
            torch.from_numpy(g_cano).float().unsqueeze(0).to(device),
        ).item()

        legacy_norms.append(n_legacy)
        legacy_cds.append(cd_legacy)
        cano_norms.append(n_cano)
        cano_cds.append(cd_cano)

        print(
            f"{tag[:70]:<70} | "
            f"{n_legacy:>15.3f}  {cd_legacy:>13.5f} | "
            f"{n_cano:>13.3f}  {cd_cano:>11.5f}"
        )

        if args.viz:
            try:
                import open3d as o3d
                from src.utils.io import to_o3d_pcd
            except Exception as e:
                print(f"  [warn] open3d 不可用，跳过可视化: {e}", file=sys.stderr)
                continue

            sep = 2.6
            blobs = [
                ("legacy.input", p_legacy, (0.95, 0.15, 0.15)),
                ("legacy.pred", pred_legacy, (1.0, 0.5, 0.05)),
                ("cano.input", p_cano, (0.75, 0.2, 0.85)),
                ("cano.pred", pred_cano, (0.15, 0.85, 0.35)),
                ("cano.gt", g_cano, (0.35, 0.45, 0.95)),
            ]
            geoms = []
            for i, (lab, pts, col) in enumerate(blobs):
                gx = (i - (len(blobs) - 1) * 0.5) * sep
                geoms.append(to_o3d_pcd(_shift_for_viz(pts, gx), color=col))
            print(
                "  viz 顺序（左→右）：红=legacy.input 橙=legacy.pred "
                "紫=cano.input 绿=cano.pred 蓝=cano.gt"
            )
            o3d.visualization.draw_geometries(
                geoms,
                window_name=f"verify cano vs legacy | {partial_pcd.stem}",
                width=1480,
                height=760,
            )

    lm, lc = np.mean(legacy_norms), np.mean(legacy_cds)
    cm, cc = np.mean(cano_norms), np.mean(cano_cds)
    mn = float(np.min(cano_norms))

    print()
    print(
        f"summary  legacy:  pred_max={lm:.3f}±{np.std(legacy_norms):.3f}"
        f"  CD-L1={lc:.5f}±{np.std(legacy_cds):.5f}"
    )
    print(
        f"summary  cano  :  pred_max={cm:.3f}±{np.std(cano_norms):.3f}"
        f"  CD-L1={cc:.5f}±{np.std(cano_cds):.5f}"
    )

    # 典型「坍缩」：预训练在错位分布上 pred_max 会远 < 0.5；PCN 预训练在 canonical 上应接近单位球 (~0.9–1.0)。
    # 不要求 cano.pred_max 一定大于 legacy（legacy 可能因尺度错位偶发 >1）。
    cano_healthy = mn >= 0.35 and cm >= 0.75
    cd_better_or_close = cc <= lc * 1.05

    if cano_healthy and cd_better_or_close:
        print(
            "\n[OK] canonical 路径：补全半径落在预训练期望范围（未坍缩），"
            "且相对 GT 的 CD-L1 不劣于 legacy（或与之一致）。"
        )
        return 0
    if not cano_healthy:
        print(
            "\n[WARN] cano 侧 pred_max 仍偏低，可能仍存在分布问题；"
            "请增大 --num-samples 或检查 ckpt / 数据。"
        )
    else:
        print(
            "\n[WARN] cano CD-L1 明显高于 legacy，可增大样本数或检查归一化。"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
