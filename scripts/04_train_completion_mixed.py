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

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
import wandb_utils  # noqa: E402

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


@torch.no_grad()
def validate(model, loader, device, max_batches=None, log_wandb_3d=False, log_step=0):
    model.eval()
    total = 0.0
    n_batch = 0
    for i, (inp, gt) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        inp, gt = inp.to(device, non_blocking=True), gt.to(device, non_blocking=True)
        outs = model(inp)
        pred = outs[-1]
        if log_wandb_3d and i == 0:
            wandb_utils.log_val_pointclouds(
                gt[0].detach().cpu().numpy(),
                pred[0].detach().cpu().numpy(),
                step=log_step,
                prefix="val",
            )
        total += chamfer_l1(pred, gt).item()
        n_batch += 1
    model.train()
    return total / max(n_batch, 1)


def main(args):
    args.stage_weights = parse_stage_weights(args.stage_weights)
    wandb_run = wandb_utils.init_wandb(args)
    try:
        _run_training(args, wandb_run)
    finally:
        wandb_utils.finish_wandb(wandb_run)


def _run_training(args, wandb_run):
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
    if len(train_ds) >= 100_000:
        log.info(
            "数据集规模较大（如 130k 级）：DataLoader 流式迭代；step 级 mini-val 已启用。"
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = SnowflakeNet(up_factors=[1, 4, 8]).to(device)
    load_pretrained(model, args.ckpt_pretrain)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=args.lr_patience,
        min_lr=1e-6,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best = float("inf")
    best_path = os.path.join(args.save_dir, "ckpt-mixed-best.pth")

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_steps_in_epoch = 0

        for i, (inp, gt) in enumerate(train_loader):
            inp, gt = inp.to(device, non_blocking=True), gt.to(device, non_blocking=True)
            optimizer.zero_grad()
            outs = model(inp)
            loss = compute_training_loss(
                outs,
                gt,
                stage_weights=args.stage_weights,
                edge_loss_weight=args.edge_loss_weight,
                edge_k=args.edge_k,
                edge_power=args.edge_power,
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_steps_in_epoch += 1
            global_step += 1

            if global_step % args.log_every == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                log.info(
                    f"Epoch {epoch:03d} step {i+1:5d}/{len(train_loader)} "
                    f"| gstep {global_step:7d} | loss {loss.item():.6f} | lr {lr_now:.2e}"
                )
                wandb_utils.log_train_metrics(loss.item(), lr_now, global_step)

            if global_step % args.val_every_steps == 0:
                val_cd = validate(
                    model,
                    val_loader,
                    device,
                    max_batches=args.val_batches,
                    log_wandb_3d=(wandb_run is not None),
                    log_step=global_step,
                )
                scheduler.step(val_cd)
                lr_now = optimizer.param_groups[0]["lr"]
                log.info(
                    f"[mini-val] gstep {global_step:7d} | val_cd {val_cd:.6f} "
                    f"| lr {lr_now:.2e} | best {best:.6f}"
                )
                wandb_utils.log_wandb_scalars(
                    {
                        "val/mini_chamfer": val_cd,
                        "val/learning_rate": lr_now,
                        "val/best_chamfer": best,
                    },
                    global_step,
                )
                if val_cd < best:
                    best = val_cd
                    torch.save(
                        {
                            "epoch": epoch,
                            "global_step": global_step,
                            "model": model.state_dict(),
                            "val_cd": val_cd,
                        },
                        best_path,
                    )
                    log.info(f"✅ new best (mini-val) val={best:.6f} saved={best_path}")

        tr = running_loss / max(n_steps_in_epoch, 1)
        va = validate(model, val_loader, device, max_batches=None)
        lr_now = optimizer.param_groups[0]["lr"]
        log.info(
            f"Epoch {epoch:03d}/{args.epochs} train {tr:.6f} full_val {va:.6f} lr {lr_now:.2e} best {best:.6f}"
        )
        if va < best:
            best = va
            torch.save(
                {"epoch": epoch, "global_step": global_step, "model": model.state_dict(), "val_cd": va},
                best_path,
            )
            log.info(f"✅ new best val={best:.6f} saved={best_path}")

        wandb_utils.log_wandb_scalars(
            {
                "epoch": epoch,
                "train/epoch_avg_loss": tr,
                "val/full_chamfer": va,
                "val/best_chamfer": best,
            },
            global_step,
        )


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
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--no-wandb", action="store_true", help="禁用 Weights & Biases")
    p.add_argument("--log-every", type=int, default=20, help="每 N 个 global step 记录 train 日志与 W&B")
    p.add_argument("--val-every-steps", type=int, default=2000)
    p.add_argument("--val-batches", type=int, default=200, help="mini-val 用前 N 个 batch")
    p.add_argument(
        "--lr-patience",
        type=int,
        default=5,
        help="ReduceLROnPlateau：连续多少个 mini-val 无改善则降 lr",
    )
    main(p.parse_args())

