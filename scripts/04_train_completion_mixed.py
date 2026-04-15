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
    return (dist1.sqrt().mean(dim=1) + dist2.sqrt().mean(dim=1)).mean()


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


def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    weights = [0.0, 0.1, 0.2, 0.7]
    for i, (inp, gt) in enumerate(loader):
        inp, gt = inp.to(device), gt.to(device)
        optimizer.zero_grad()
        outs = model(inp)
        loss = sum(w * chamfer_l1(o, gt) for w, o in zip(weights, outs))
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"device={device}")

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
        tr = train_one_epoch(model, train_loader, optimizer, device, epoch)
        va = validate(model, val_loader, device)
        scheduler.step(va)
        log.info(f"Epoch {epoch:03d}/{args.epochs} train {tr:.6f} val {va:.6f} lr {optimizer.param_groups[0]['lr']:.2e}")
        if va < best:
            best = va
            torch.save({"epoch": epoch, "model": model.state_dict(), "val_cd": va}, best_path)
            log.info(f"✅ new best val={best:.6f} saved={best_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--modelnet_root", default="/home/csy/SnowflakeNet_FPFH_ICP/data/processed_with_removal")
    p.add_argument("--itodd_root", default="/home/csy/SnowflakeNet_FPFH_ICP/data/itodd_processed_with_removal")
    p.add_argument("--ckpt_pretrain", default="/home/csy/SnowflakeNet_FPFH_ICP/Snet/SnowflakeNet-main/completion/checkpoints/ckpt-best-pcn-cd_l1.pth")
    p.add_argument("--save_dir", default="/home/csy/SnowflakeNet_FPFH_ICP/checkpoints/snet_finetune_mixed")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    main(p.parse_args())

