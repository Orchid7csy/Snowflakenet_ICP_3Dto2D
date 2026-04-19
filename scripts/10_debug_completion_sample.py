"""
单样本补全诊断脚本。

用途：
1) 量化 input -> gt 的原生 Chamfer-L1，判断数据本身的“信噪比”。
2) 量化 SnowflakeNet 各阶段输出 [Pc, P1, P2, P3] 的 CD，判断是 seed 先歪，还是细化阶段失效。
3) 用局部协方差最小特征值占比估计曲率，抽取 GT 边缘点，统计各阶段对边缘的覆盖率。
4) 导出 input / gt / pc_seed / p1 / p2 / p3 / gt_edge，便于用 view_npy 或 Open3D 直接看。

示例：
    python scripts/10_debug_completion_sample.py \
        --data-root data/processed/itodd \
        --split test \
        --stem obj000001_aug0031_v0 \
        --ckpt checkpoints/snet_finetune_itodd/ckpt-itodd-best.pth \
        --export-dir debug/sample_case \
        --vis
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import OrderedDict
from typing import Dict, List

import numpy as np
import open3d as o3d
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_SNET_ROOT = os.path.join(_PROJECT_ROOT, "Snet", "SnowflakeNet-main")

for _p in (_PROJECT_ROOT, _SNET_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from loss_functions.Chamfer3D.dist_chamfer_3D import chamfer_3DDist
from models.model_completion import SnowflakeNet

_CHAMFER = chamfer_3DDist()


def _resolve_data_paths(data_root: str, split: str, stem: str) -> Dict[str, str]:
    split_root = os.path.join(os.path.abspath(data_root), split)
    return {
        "input": os.path.join(split_root, "input", f"{stem}.npy"),
        "gt": os.path.join(split_root, "gt", f"{stem}.npy"),
    }


def _resample_points(points: np.ndarray, target_n: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] == target_n:
        return points.astype(np.float32)
    if points.shape[0] > target_n:
        idx = rng.choice(points.shape[0], target_n, replace=False)
    else:
        idx = rng.choice(points.shape[0], target_n, replace=True)
    return points[idx].astype(np.float32)


def _load_model(ckpt_path: str) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_loc = device if device.type == "cuda" else "cpu"
    model = SnowflakeNet(up_factors=[1, 4, 8])
    checkpoint = torch.load(ckpt_path, map_location=map_loc)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        name = key[7:] if key.startswith("module.") else key
        new_state_dict[name] = value

    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


def _chamfer_l1_cpu_fallback(pred: np.ndarray, gt: np.ndarray) -> Dict[str, np.ndarray | float]:
    """
    纯 CPU 后备：O(N^2) 的 cdist Chamfer（仅用于诊断，点数大时会慢）。
    返回的 dist 是欧氏距离（已开方），与 Chamfer-L1 定义一致。
    """
    pred_t = torch.from_numpy(pred).float().unsqueeze(0)  # (1,N,3)
    gt_t = torch.from_numpy(gt).float().unsqueeze(0)      # (1,M,3)
    with torch.no_grad():
        d = torch.cdist(pred_t, gt_t)  # (1,N,M)
        d1 = d.min(dim=2).values[0].cpu().numpy()  # pred->gt
        d2 = d.min(dim=1).values[0].cpu().numpy()  # gt->pred
    return {
        "cd_l1": float(d1.mean() + d2.mean()),
        "pred_to_gt_l1": float(d1.mean()),
        "gt_to_pred_l1": float(d2.mean()),
        "gt_nn_l1": d2,
    }


def _chamfer_l1_with_details(pred: np.ndarray, gt: np.ndarray, device: torch.device) -> Dict[str, np.ndarray | float]:
    pred_t = torch.from_numpy(pred).float().unsqueeze(0).to(device)
    gt_t = torch.from_numpy(gt).float().unsqueeze(0).to(device)
    try:
        with torch.no_grad():
            dist1, dist2, _, _ = _CHAMFER(pred_t, gt_t)
        d1 = (dist1[0] + 1e-8).sqrt().detach().cpu().numpy()
        d2 = (dist2[0] + 1e-8).sqrt().detach().cpu().numpy()
        return {
            "cd_l1": float(d1.mean() + d2.mean()),
            "pred_to_gt_l1": float(d1.mean()),
            "gt_to_pred_l1": float(d2.mean()),
            "gt_nn_l1": d2,
        }
    except Exception as e:
        # Chamfer3D 官方实现通常是 CUDA-only；若传入 CPU tensor 会在内部 set_device 处炸。
        if device.type == "cpu":
            raise
        print(f"[warn] Chamfer3D CUDA 失败，切换 CPU fallback（更慢）。错误: {e}")
        return _chamfer_l1_cpu_fallback(pred, gt)


def _estimate_curvature_scores(points: np.ndarray, k: int) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    n = pts.shape[0]
    if n == 0:
        return np.zeros((0,), dtype=np.float32)
    if n == 1:
        return np.ones((1,), dtype=np.float32)

    k = int(max(1, min(k, n - 1)))
    diff = pts[:, None, :] - pts[None, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    knn_idx = np.argpartition(dist2, kth=k, axis=1)[:, 1 : k + 1]
    neigh = pts[knn_idx]
    centered = neigh - pts[:, None, :]
    cov = np.matmul(np.transpose(centered, (0, 2, 1)), centered) / float(k)
    eigvals = np.linalg.eigvalsh(cov)
    scores = eigvals[:, 0] / np.maximum(eigvals.sum(axis=1), 1e-12)
    return scores.astype(np.float32)


def _threshold_list(text: str) -> List[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if not vals:
        raise ValueError("thresholds 不能为空")
    return vals


def _stage_name_list(n_stage: int) -> List[str]:
    default = ["pc_seed", "p1", "p2", "p3"]
    if n_stage <= len(default):
        return default[:n_stage]
    return default + [f"stage_{i}" for i in range(len(default), n_stage)]


def _save_arrays(export_dir: str, stem: str, arrays: Dict[str, np.ndarray]) -> Dict[str, str]:
    os.makedirs(export_dir, exist_ok=True)
    saved = {}
    for name, arr in arrays.items():
        path = os.path.join(export_dir, f"{stem}_{name}.npy")
        np.save(path, arr.astype(np.float32))
        saved[name] = path
    return saved


def _pcd(points: np.ndarray, color: List[float], translate=None) -> o3d.geometry.PointCloud:
    out = o3d.geometry.PointCloud()
    pts = np.asarray(points, dtype=np.float64)
    if translate is not None:
        pts = pts + np.asarray(translate, dtype=np.float64).reshape(1, 3)
    out.points = o3d.utility.Vector3dVector(pts)
    out.paint_uniform_color(color)
    return out


def _visualize(input_pts: np.ndarray, gt: np.ndarray, gt_edge: np.ndarray, outputs: Dict[str, np.ndarray]) -> None:
    clouds = [
        _pcd(input_pts, [1.0, 0.1, 0.1], translate=[0.0, 0.0, 0.0]),
        _pcd(gt, [0.1, 0.9, 0.1], translate=[1.6, 0.0, 0.0]),
        _pcd(gt_edge, [1.0, 0.0, 1.0], translate=[1.6, 0.0, 0.0]),
    ]
    offsets = {
        "pc_seed": [3.2, 0.0, 0.0],
        "p1": [4.8, 0.0, 0.0],
        "p2": [6.4, 0.0, 0.0],
        "p3": [8.0, 0.0, 0.0],
    }
    colors = {
        "pc_seed": [1.0, 0.8, 0.1],
        "p1": [0.2, 0.6, 1.0],
        "p2": [0.0, 0.8, 0.8],
        "p3": [0.0, 0.6, 0.2],
    }
    for name, pts in outputs.items():
        clouds.append(_pcd(pts, colors.get(name, [0.5, 0.5, 0.5]), translate=offsets.get(name, [0.0, 0.0, 0.0])))
    o3d.visualization.draw_geometries(
        clouds,
        window_name="completion debug: input | gt(+edge) | pc_seed | p1 | p2 | p3",
        width=1600,
        height=900,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="单样本补全调试")
    parser.add_argument("--data-root", default=os.path.join(_PROJECT_ROOT, "data", "processed", "itodd"))
    parser.add_argument("--split", default="test", choices=("train", "test"))
    parser.add_argument("--stem", required=True, help="样本主文件名，不带扩展名")
    parser.add_argument(
        "--ckpt",
        default=os.path.join(_PROJECT_ROOT, "checkpoints", "snet_finetune_itodd", "ckpt-itodd-best.pth"),
    )
    parser.add_argument("--num-points", type=int, default=2048, help="模型输入点数")
    parser.add_argument("--edge-k", type=int, default=24, help="曲率估计的 kNN 邻域大小")
    parser.add_argument("--edge-percentile", type=float, default=85.0, help="取曲率最高多少分位以上为边缘点")
    parser.add_argument(
        "--edge-thresholds",
        default="0.01,0.02,0.04",
        help="统计 edge recall 的距离阈值，逗号分隔",
    )
    parser.add_argument(
        "--export-dir",
        default=os.path.join(_PROJECT_ROOT, "debug", "completion_sample"),
        help="导出 input/gt/stages/summary 的目录",
    )
    parser.add_argument("--seed", type=int, default=42, help="重采样随机种子，保证可复现")
    parser.add_argument("--vis", action="store_true", help="弹出 Open3D 可视化窗口")
    args = parser.parse_args()

    paths = _resolve_data_paths(args.data_root, args.split, args.stem)
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} 文件不存在: {path}")
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"checkpoint 不存在: {args.ckpt}")

    rng = np.random.default_rng(args.seed)
    input_raw = np.load(paths["input"]).astype(np.float32)
    gt = np.load(paths["gt"]).astype(np.float32)
    input_pts = _resample_points(input_raw, args.num_points, rng)

    model = _load_model(args.ckpt)
    device = next(model.parameters()).device
    with torch.no_grad():
        outputs = model(torch.from_numpy(input_pts).unsqueeze(0).to(device))
    stage_names = _stage_name_list(len(outputs))
    stage_outputs = {
        name: tensor.squeeze(0).detach().cpu().numpy().astype(np.float32)
        for name, tensor in zip(stage_names, outputs)
    }

    curvature = _estimate_curvature_scores(gt, k=args.edge_k)
    edge_thr = float(np.percentile(curvature, args.edge_percentile))
    edge_mask = curvature >= edge_thr
    gt_edge = gt[edge_mask]
    edge_thresholds = _threshold_list(args.edge_thresholds)

    report = {
        "stem": args.stem,
        "split": args.split,
        "input_points": int(input_pts.shape[0]),
        "gt_points": int(gt.shape[0]),
        "edge_points": int(gt_edge.shape[0]),
        "edge_percentile": float(args.edge_percentile),
        "edge_threshold": edge_thr,
        "metrics": {},
    }

    # Chamfer3D 是 CUDA-only：这里用与模型同一个 device，避免误用 CPU tensor
    baseline = _chamfer_l1_with_details(input_pts, gt, device=device)
    report["metrics"]["input"] = {
        "cd_l1": baseline["cd_l1"],
        "pred_to_gt_l1": baseline["pred_to_gt_l1"],
        "gt_to_pred_l1": baseline["gt_to_pred_l1"],
    }

    for name, pts in stage_outputs.items():
        metrics = _chamfer_l1_with_details(pts, gt, device=device)
        edge_nn = np.asarray(metrics["gt_nn_l1"])[edge_mask]
        edge_recall = {
            f"edge_recall@{thr:g}": float((edge_nn <= thr).mean()) if edge_nn.size else 0.0
            for thr in edge_thresholds
        }
        report["metrics"][name] = {
            "cd_l1": metrics["cd_l1"],
            "pred_to_gt_l1": metrics["pred_to_gt_l1"],
            "gt_to_pred_l1": metrics["gt_to_pred_l1"],
            "edge_gt_to_pred_l1": float(edge_nn.mean()) if edge_nn.size else 0.0,
            **edge_recall,
        }

    print("=== Completion Debug Report ===")
    print(f"sample      : {args.stem}")
    print(f"split       : {args.split}")
    print(f"input->gt CD: {report['metrics']['input']['cd_l1']:.6f}")
    print(f"edge points : {report['edge_points']} / {report['gt_points']}  (p{args.edge_percentile:.1f}+)")
    print("")
    print(f"{'stage':<10} {'CD-L1':>12} {'pred->gt':>12} {'gt->pred':>12} {'edge gt->pred':>16}")
    print("-" * 66)
    print(
        f"{'input':<10} "
        f"{report['metrics']['input']['cd_l1']:>12.6f} "
        f"{report['metrics']['input']['pred_to_gt_l1']:>12.6f} "
        f"{report['metrics']['input']['gt_to_pred_l1']:>12.6f} "
        f"{'-':>16}"
    )
    for name in stage_names:
        item = report["metrics"][name]
        print(
            f"{name:<10} "
            f"{item['cd_l1']:>12.6f} "
            f"{item['pred_to_gt_l1']:>12.6f} "
            f"{item['gt_to_pred_l1']:>12.6f} "
            f"{item['edge_gt_to_pred_l1']:>16.6f}"
        )
    print("")
    for name in stage_names:
        item = report["metrics"][name]
        recalls = "  ".join(
            f"recall@{thr:g}={item[f'edge_recall@{thr:g}']:.3f}" for thr in edge_thresholds
        )
        print(f"{name:<10} {recalls}")

    export_dir = os.path.abspath(args.export_dir)
    arrays = {
        "input": input_pts,
        "gt": gt,
        "gt_edge": gt_edge,
        "gt_curvature": curvature.reshape(-1, 1),
        **stage_outputs,
    }
    saved = _save_arrays(export_dir, args.stem, arrays)
    summary_path = os.path.join(export_dir, f"{args.stem}_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("")
    print(f"导出目录    : {export_dir}")
    print(f"summary json : {summary_path}")
    for name, path in saved.items():
        print(f"{name:<12}: {path}")

    if args.vis:
        _visualize(input_pts, gt, gt_edge, stage_outputs)


if __name__ == "__main__":
    main()
