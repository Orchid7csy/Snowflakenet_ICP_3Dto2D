"""
scripts/03_train_completion_itodd.py

基于 scripts/02_train_completion.py 的 ITODD 版本：
- 只需要 data_root 下的 split/{input,gt}（obs/meta 会被忽略）
- 默认 data_root 指向 data/itodd_processed_with_removal
"""

import csv
import os
import sys
import argparse
import logging
from collections import OrderedDict
from datetime import datetime
from typing import List

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

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


def chamfer_l1(pred: torch.Tensor, gt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    dist1, dist2, _, _ = _chamfer_fn(pred, gt)
    return ((dist1 + eps).sqrt().mean(dim=1) + (dist2 + eps).sqrt().mean(dim=1)).mean()


def parse_stage_weights(text: str) -> List[float]:
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) != 4:
        raise argparse.ArgumentTypeError("stage_weights 必须是 4 个逗号分隔浮点数，对应 Pc,P1,P2,P3")
    return vals


@torch.no_grad()
def compute_gt_edge_weights(gt: torch.Tensor, k: int = 24, power: float = 1.0) -> torch.Tensor:
    """
    用 GT 局部协方差的最小特征值占比近似“曲率/边缘性”，
    只给 gt->pred 方向额外加权，鼓励模型覆盖尖边与高曲率区域。
    """
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


def edge_aware_gt_penalty(
    pred: torch.Tensor,
    gt: torch.Tensor,
    gt_edge_weights: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    _, dist2, _, _ = _chamfer_fn(pred, gt)
    gt_to_pred = (dist2 + eps).sqrt()
    return (gt_to_pred * gt_edge_weights).mean()


def compute_training_loss(
    outputs,
    gt: torch.Tensor,
    stage_weights: List[float],
    edge_loss_weight: float,
    edge_k: int,
    edge_power: float,
) -> torch.Tensor:
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


class SnowflakeDatasetSeparateRoots(SnowflakeDataset):
    """
    input 和 gt 分别来自不同 root。
    结构要求：
      <input_root>/<split>/input/*.npy
      <gt_root>/<split>/gt/*.npy
    """

    def __init__(self, input_root: str, gt_root: str, split: str = "train", num_points: int = 2048, transform=None):
        self.input_path = os.path.join(input_root, split, "input")
        self.gt_path = os.path.join(gt_root, split, "gt")
        self.file_list = [f for f in os.listdir(self.input_path) if f.endswith(".npy")]
        self.num_points = num_points
        self.transform = transform
        self.split = split

    def _resample(self, pcd, n):
        curr_n = pcd.shape[0]
        if curr_n == n:
            return pcd
        if curr_n > n:
            idx = np.random.choice(curr_n, n, replace=False)
        else:
            if curr_n == 0:
                return np.zeros((n, 3), dtype=np.float32)
            idx = np.concatenate([np.arange(curr_n), np.random.choice(curr_n, n - curr_n, replace=True)])
        return pcd[idx]


def build_dataset(args, split: str):
    if args.input_root and args.gt_root:
        return SnowflakeDatasetSeparateRoots(
            args.input_root,
            args.gt_root,
            split=split,
            num_points=args.num_points,
        )
    return SnowflakeDataset(args.data_root, split=split, num_points=args.num_points)


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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        total_loss += loss.item()
        if (i + 1) % 20 == 0:
            log.info(f"Epoch {epoch:03d} step {i+1:4d}/{len(loader)} loss {loss.item():.6f}")
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


def plot_curves(csv_path: str, save_dir: str):
    """读取训练日志 CSV，画 loss 和 lr 曲线并保存为 PNG。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs, train_losses, val_losses, lrs = [], [], [], []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            epochs.append(int(row["epoch"]))
            train_losses.append(float(row["train_loss"]))
            val_losses.append(float(row["val_loss"]))
            lrs.append(float(row["lr"]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, "b-o", markersize=3, label="Train CD-L1")
    ax1.plot(epochs, val_losses, "r-o", markersize=3, label="Val CD-L1")
    best_idx = int(np.argmin(val_losses))
    ax1.axvline(epochs[best_idx], color="green", linestyle="--", alpha=0.6, label=f"Best val @ ep{epochs[best_idx]}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Chamfer-L1")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, lrs, "g-o", markersize=3)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("LR Schedule")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    png_path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(png_path, dpi=150)
    plt.close()
    log.info(f"曲线图已保存: {png_path}")


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

    train_ds = build_dataset(args, split="train")
    val_ds = build_dataset(args, split="test")
    log.info(f"train={len(train_ds)} val={len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = SnowflakeNet(up_factors=[1, 4, 8]).to(device)
    load_pretrained(model, args.ckpt_pretrain)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-6)

    os.makedirs(args.save_dir, exist_ok=True)
    best = float("inf")
    best_path = os.path.join(args.save_dir, "ckpt-itodd-best.pth")

    # ── CSV log ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(args.save_dir, f"train_log_{timestamp}.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch", "train_loss", "val_loss", "lr", "best_val", "is_best"])
    log.info(f"训练日志: {csv_path}")

    # ── file log ──
    file_handler = logging.FileHandler(os.path.join(args.save_dir, f"train_{timestamp}.log"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
    log.addHandler(file_handler)

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
        lr_now = optimizer.param_groups[0]["lr"]
        is_best = va < best
        if is_best:
            best = va
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_cd": va}, best_path)
            log.info(f"✅ new best val={best:.6f} saved={best_path}")

        log.info(f"Epoch {epoch:03d}/{args.epochs} train {tr:.6f} val {va:.6f} lr {lr_now:.2e} best {best:.6f}")
        csv_writer.writerow([epoch, f"{tr:.8f}", f"{va:.8f}", f"{lr_now:.2e}", f"{best:.8f}", int(is_best)])
        csv_file.flush()

    csv_file.close()
    log.info(f"训练完成。CSV 日志: {csv_path}")

    try:
        plot_curves(csv_path, args.save_dir)
    except Exception as e:
        log.warning(f"画图失败（不影响训练结果）: {e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/home/csy/SnowflakeNet_FPFH_ICP/data/processed/itodd")
    p.add_argument("--input_root", default=None, help="可选：input 的 root（若提供则覆盖 data_root）")
    p.add_argument("--gt_root", default=None, help="可选：gt 的 root（若提供则覆盖 data_root）")
    p.add_argument("--ckpt_pretrain", default="/home/csy/SnowflakeNet_FPFH_ICP/Snet/SnowflakeNet-main/completion/checkpoints/ckpt-best-pcn-cd_l1.pth")
    p.add_argument("--save_dir", default="/home/csy/SnowflakeNet_FPFH_ICP/checkpoints/snet_finetune_itodd")
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
    p.add_argument("--num_points", type=int, default=2048)
    main(p.parse_args())

