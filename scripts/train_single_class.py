"""
scripts/02_train_completion.py

SnowflakeNet 微调脚本
用法:
    python scripts/02_train_completion.py \
        --data_root data/processed \
        --ckpt_pretrain Snet/SnowflakeNet-main/completion/checkpoints/ckpt-best-pcn-cd_l1.pth \
        --save_dir checkpoints/snet_finetune \
        --epochs 100 \
        --batch_size 8 \
        --lr 1e-4
"""

import os
import sys
import argparse
import logging
from collections import OrderedDict

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# ── 路径设置，让脚本从项目根目录运行 ──────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
snet_root    = os.path.join(project_root, 'Snet', 'SnowflakeNet-main')

# 注意这里：直接将 snet_root 加入 sys.path，而不是 snet_root/completion
for p in [project_root, snet_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.data.dataset import SnowflakeDataset          # 你写的 Dataset
from models.model_completion import SnowflakeNet        # 官方模型
from loss_functions.Chamfer3D.dist_chamfer_3D import chamfer_3DDist  # 官方 Chamfer


# ── 日志 ──────────────────────────────────────────────────────────────────
# ── 日志配置 ──────────────────────────────────────────────────────────────────
def setup_logging(save_dir):
    log_file = os.path.join(save_dir, 'train.log')
    os.makedirs(save_dir, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


# ── Chamfer Distance L1（与官方预训练一致） ───────────────────────────────
_chamfer_fn = chamfer_3DDist()

def chamfer_l1(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    pred, gt: (B, N, 3)
    返回标量，单位与坐标单位一致
    """
    dist1, dist2, _, _ = _chamfer_fn(pred, gt)
    # L1 版本：取 sqrt 再平均
    cd = ((dist1 + 1e-8).sqrt().mean(dim=1) + (dist2 + 1e-8).sqrt().mean(dim=1)).mean()
    return cd


# ── 加载预训练权重（去 module. 前缀） ─────────────────────────────────────
def load_pretrained(model: SnowflakeNet, ckpt_path: str, log): 
    checkpoint  = torch.load(ckpt_path, map_location='cpu')
    state_dict  = checkpoint['model']

    new_sd = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_sd[name] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=True)
    assert not missing,    f"Missing keys: {missing}"
    assert not unexpected, f"Unexpected keys: {unexpected}"
    # 🚨 这里现在可以使用传入的 log 了
    log.info(f"预训练权重加载成功：{ckpt_path}")


# ── 单 epoch 训练 ──────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, device, epoch, log):
    model.train()
    total_loss = 0.0

    for i, (inp, gt) in enumerate(loader):
        # inp, gt: (B, N, 3)
        inp, gt = inp.to(device), gt.to(device)

        optimizer.zero_grad()

        # SnowflakeNet 返回 [Pc, P1, P2, P3]
        # Pc 是种子点（粗粒度），P1/P2/P3 是三个 SPD 阶段输出
        outputs = model(inp)

        # 对所有中间输出都算损失，权重递增（粗→细）
        # 这与官方训练策略一致，有助于稳定早期训练
        weights = [1.0, 1.0, 1.0, 1.0]   # 对应 Pc, P1, P2, P3
        loss = sum(w * chamfer_l1(out, gt) for w, out in zip(weights, outputs))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (i + 1) % 20 == 0:
            log.info(f"  Epoch {epoch:03d} | step {i+1:4d}/{len(loader)} "
                     f"| loss {loss.item():.6f}")

    return total_loss / len(loader)


# ── 验证 ───────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    total_cd = 0.0

    for inp, gt in loader:
        inp, gt = inp.to(device), gt.to(device)
        outputs = model(inp)
        pred    = outputs[-1]          # 只取最终输出 P3 评估
        total_cd += chamfer_l1(pred, gt).item()

    return total_cd / len(loader)


# ── 主函数 ─────────────────────────────────────────────────────────────────
def main(args):
    log = setup_logging(args.save_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"使用设备：{device}")

    # 🚨 单类别验证模式
    target_category = 'airplane'
    log.info(f"🚀 模式：单类别验证 [{target_category}] (包含多视角增强)")

    # 数据加载与过滤
    train_ds = SnowflakeDataset(args.data_root, split='train', num_points=2048)
    val_ds   = SnowflakeDataset(args.data_root, split='test', num_points=2048)
    
    # 根据文件名过滤目标类别 (适配 8 视角文件名)
    train_ds.file_list = [f for f in train_ds.file_list if target_category in f]
    val_ds.file_list   = [f for f in val_ds.file_list if target_category in f]
    
    log.info(f"📊 过滤后：训练集 {len(train_ds)} 样本 | 验证集 {len(val_ds)} 样本")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, 
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, 
                              num_workers=4, pin_memory=True)

    # 模型初始化
    model = SnowflakeNet(up_factors=[1, 4, 8]).to(device)
    load_pretrained(model, args.ckpt_pretrain,log)

    # 优化器与微调调度器
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6
    )

    best_cd = float('inf')
    best_path = os.path.join(args.save_dir, f'ckpt-{target_category}-best.pth')

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, epoch, log)
        val_cd     = validate(model, val_loader, device)
        
        # 根据验证集表现调整学习率
        scheduler.step(val_cd) 

        # 修正：从 optimizer 读取当前学习率进行打印
        curr_lr = optimizer.param_groups[0]['lr']
        log.info(f"Epoch {epoch:03d}/{args.epochs} | train_loss {train_loss:.6f} | "
                 f"val_cd {val_cd:.6f} | lr {curr_lr:.2e}")

        # 保存带类别标签的最优模型
        if val_cd < best_cd:
            best_cd = val_cd
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'val_cd': val_cd}, best_path)
            log.info(f"  ✅ 新最优 [{target_category}] val_cd={best_cd:.6f}，已保存")

        # 每 10 轮保存常规备份
        if epoch % 10 == 0:
            reg_path = os.path.join(args.save_dir, f'ckpt-{target_category}-epoch{epoch:03d}.pth')
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'val_cd': val_cd}, reg_path)

    log.info(f"训练结束。[{target_category}] 最优 val_cd = {best_cd:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # ── 修改处：添加默认路径并移除 required=True ──────────────────────────
    parser.add_argument('--data_root',     type=str, 
                        default='/home/csy/SnowflakeNet_FPFH_ICP/data/processed_with_removal',
                        help='data/processed 的路径')
    parser.add_argument('--ckpt_pretrain', type=str, 
                        default='/home/csy/SnowflakeNet_FPFH_ICP/Snet/SnowflakeNet-main/completion/checkpoints/ckpt-best-pcn-cd_l1.pth',
                        help='预训练权重路径 (.pth)')
    # ──────────────────────────────────────────────────────────────────────
    parser.add_argument('--save_dir',      type=str, default='/home/csy/SnowflakeNet_FPFH_ICP/checkpoints/snet_finetune')
    parser.add_argument('--epochs',        type=int, default=100)
    parser.add_argument('--batch_size',    type=int, default=8)
    parser.add_argument('--lr',            type=float, default=1e-5)
    args = parser.parse_args()
    main(args)