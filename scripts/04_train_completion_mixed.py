"""
scripts/04_train_completion_mixed.py

混合 ModelNet40 + ITODD 两类数据进行 SnowflakeNet completion 微调。
两套数据都要求遵循同样的 io 结构：
  <root>/<split>/{input,gt}/*.npy

策略：
- ConcatDataset 合并
- DataLoader shuffle
（如果你想严格按比例采样，可进一步换成 WeightedRandomSampler）
"""

import os
import sys
import argparse
import logging
from collections import OrderedDict
from typing import List

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
snet_root = os.path.join(project_root, "Snet", "SnowflakeNet-main")
for p in [project_root, snet_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data.dataset import SnowflakeDataset
from models.model_completion import SnowflakeNet
from loss_functions.Chamfer3D.dist_chamfer_3D import chamfer_3DDist

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

_chamfer_fn = chamfer_3DDist()


def chamfer_l1(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    dist1, dist2, _, _ = _chamfer_fn(pred, gt)
    return ((dist1 + 1e-8).sqrt().mean(dim=1) + (dist2 + 1e-8).sqrt().mean(dim=1)).mean()


def parse_stage_weights(text: str) -> List[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("stage_weights 必须是 4 个逗号分隔浮点数，对应 Pc,P1,P2,P3")
    return vals


@torch.no_grad()
def compute_gt_edge_weights(gt: torch.Tensor, k: int = 24, power: float = 1.0) -> torch.Tensor:
    bsz, n_pts, _ = gt.shape
    if n_pts <= 1:
        return torch.ones((bsz, n_pts), device=gt.device, dtype=gt.dtype)

    k = max(1, min(int(k), n_pts - 1))
    pairwise = torch.cdist(gt, gt)
    knn_idx = pairwise.topk(k=k + 1, largest=False).indices[:, :, 1:]
    batch_idx = torch.arange(bsz, device=gt.device).view(bsz, 1, 1)
    neighbors = gt[batch_idx, knn_idx]
    centered = neighbors - gt.unsqueeze(2)
    cov = centered.transpose(-1, -2) @ centered / float(k)
    eigvals = torch.linalg.eigvalsh(cov)
    curvature = eigvals[..., 0] / eigvals.sum(dim=-1).clamp_min(1e-8)
    weights = curvature.clamp_min(1e-6).pow(float(power))
    weights = weights / weights.mean(dim=1, keepdim=True).clamp_min(1e-8)
    return weights


def edge_aware_gt_penalty(pred: torch.Tensor, gt: torch.Tensor, gt_edge_weights: torch.Tensor) -> torch.Tensor:
    _, dist2, _, _ = _chamfer_fn(pred, gt)
    gt_to_pred = (dist2 + 1e-8).sqrt()
    return (gt_to_pred * gt_edge_weights).mean()


def compute_training_loss(outputs, gt: torch.Tensor, stage_weights: List[float], edge_loss_weight: float, edge_k: int, edge_power: float) -> torch.Tensor:
    gt_edge_weights = None
    if edge_loss_weight > 0:
        gt_edge_weights = compute_gt_edge_weights(gt, k=edge_k, power=edge_power)

    total = torch.zeros((), device=gt.device, dtype=gt.dtype)
    for stage_w, out in zip(stage_weights, outputs):
        if stage_w == 0:
            continue
        stage_loss = chamfer_l1(out, gt)
        if gt_edge_weights is not None:
            stage_loss = stage_loss + float(edge_loss_weight) * edge_aware_gt_penalty(out, gt, gt_edge_weights)
        total = total + float(stage_w) * stage_loss
    return total


def load_pretrained(model: SnowflakeNet, ckpt_path: str):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["model"]
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_sd[name] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=True)
    assert not missing and not unexpected
    log.info(f"预训练权重加载成功：{ckpt_path}")


def train_one_epoch(model, loader, optimizer, device, epoch, stage_weights, edge_loss_weight, edge_k, edge_power):
    model.train()
    total_loss = 0.0
    for i, (inp, gt) in enumerate(loader):
        inp, gt = inp.to(device), gt.to(device)
        optimizer.zero_grad()
        outs = model(inp)
        loss = compute_training_loss(
            outs,
            gt,
            stage_weights=stage_weights,
            edge_loss_weight=edge_loss_weight,
            edge_k=edge_k,
            edge_power=edge_power,
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % 50 == 0:
            log.info(f"Epoch {epoch:03d} step {i+1:5d}/{len(loader)} loss {loss.item():.6f}")
    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total = 0.0
    for inp, gt in loader:
        inp, gt = inp.to(device), gt.to(device)
        pred = model(inp)[-1]
        total += chamfer_l1(pred, gt).item()
    return total / len(loader)


def main(args):
    args.stage_weights = parse_stage_weights(args.stage_weights)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device={device}")
    log.info(
        "loss config | stage_weights=%s edge_loss_weight=%.4f edge_k=%d edge_power=%.3f",
        args.stage_weights,
        args.edge_loss_weight,
        args.edge_k,
        args.edge_power,
    )

    mn_train = SnowflakeDataset(args.modelnet_root, split="train", num_points=2048)
    mn_val = SnowflakeDataset(args.modelnet_root, split="test", num_points=2048)
    it_train = SnowflakeDataset(args.itodd_root, split="train", num_points=2048)
    it_val = SnowflakeDataset(args.itodd_root, split="test", num_points=2048)

    train_ds = ConcatDataset([mn_train, it_train])
    val_ds = ConcatDataset([mn_val, it_val])
    log.info(f"train total={len(train_ds)} (mn={len(mn_train)} it={len(it_train)})")
    log.info(f"val   total={len(val_ds)} (mn={len(mn_val)} it={len(it_val)})")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = SnowflakeNet(up_factors=[1, 4, 8]).to(device)
    load_pretrained(model, args.ckpt_pretrain)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)

    os.makedirs(args.save_dir, exist_ok=True)
    best = float("inf")
    best_path = os.path.join(args.save_dir, "ckpt-mixed-best.pth")

    for epoch in range(1, args.epochs + 1):
        tr = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epoch,
            stage_weights=args.stage_weights,
            edge_loss_weight=args.edge_loss_weight,
            edge_k=args.edge_k,
            edge_power=args.edge_power,
        )
        va = validate(model, val_loader, device)
        scheduler.step(va)
        log.info(f"Epoch {epoch:03d}/{args.epochs} train {tr:.6f} val {va:.6f} lr {optimizer.param_groups[0]['lr']:.2e}")
        if va < best:
            best = va
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_cd": va}, best_path)
            log.info(f"✅ new best val={best:.6f} saved={best_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--modelnet_root", default="/home/csy/SnowflakeNet_FPFH_ICP/data/processed/modelnet40/processed_with_removal_16k_min1024")
    p.add_argument("--itodd_root", default="/home/csy/SnowflakeNet_FPFH_ICP/data/processed/itodd")
    p.add_argument("--ckpt_pretrain", default="/home/csy/SnowflakeNet_FPFH_ICP/Snet/SnowflakeNet-main/completion/checkpoints/ckpt-best-pcn-cd_l1.pth")
    p.add_argument("--save_dir", default="/home/csy/SnowflakeNet_FPFH_ICP/checkpoints/snet_finetune_mixed")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument(
        "--stage_weights",
        type=str,
        default="0.0,0.1,0.2,0.7",
        help="四阶段 loss 权重，顺序为 Pc,P1,P2,P3",
    )
    p.add_argument(
        "--edge_loss_weight",
        type=float,
        default=0.0,
        help="额外的 edge-aware gt->pred 惩罚权重；0 表示关闭",
    )
    p.add_argument("--edge_k", type=int, default=24, help="edge-aware 权重的 kNN 邻域大小")
    p.add_argument("--edge_power", type=float, default=1.0, help="edge-aware 曲率权重指数")
    main(p.parse_args())

