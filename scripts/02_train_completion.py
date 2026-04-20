"""
scripts/02_train_completion.py

SnowflakeNet 微调脚本（与原 PCN/SnowflakeNet 实验对齐版本）

关键修复（相对旧版）：
  1) GT=16384 与 up_factors=[1,4,8] 同分辨率（P3 也是 16384），消除“稀疏 pred ↔ 密集 GT”
     的 Chamfer 跨分辨率退化。Pc/P1/P2 用 FPS 风格的随机子采样，把 GT 降到与各 stage 相同
     的点数后再算 CD。
  2) Step 级验证 / 调度 / 保存（默认每 2000 step 一次 mini-val），Epoch 级仅用作汇总日志。
  3) 维度自检：第一次 forward 打印各 stage 形状，确认 P3 == GT_points。

注意：
  - 必须重新预处理（gt 目录里的 .npy 必须是 16384 点）。运行
        python scripts/01_preprocess_modelnet40.py --num-gt 16384
    （脚本默认值已改为 16384）。

用法：
    python scripts/02_train_completion.py

W&B：默认 project=`SnowflakeNet_Finetune`，可用环境变量 `WANDB_*` 配置；`--no-wandb` 仅保留本地日志与 checkpoint。
"""

import argparse
import logging
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# ── 路径设置 ────────────────────────────────────────────────────────────
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
snet_root = os.path.join(project_root, 'Snet', 'SnowflakeNet-main')
log_dir = '/home/csy/SnowflakeNet_FPFH_ICP/checkpoints/snet_finetune'
os.makedirs(log_dir, exist_ok=True)
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


# ── 日志 ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S',
)
file_handler = logging.FileHandler(
    os.path.join(log_dir, 'train_all_classes.log'),
    mode='w',
    encoding='utf-8',
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(
    logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
)
logging.getLogger().addHandler(file_handler)
log = logging.getLogger(__name__)


# ── Chamfer Distance L1（与官方预训练一致） ───────────────────────────
_chamfer_fn = chamfer_3DDist()


def chamfer_l1(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """对称 L1 CD。pred/gt: (B, N, 3) / (B, M, 3)。"""
    dist1, dist2, _, _ = _chamfer_fn(pred, gt)
    cd = ((dist1 + 1e-8).sqrt().mean(dim=1) + (dist2 + 1e-8).sqrt().mean(dim=1)).mean()
    return cd


def random_subsample_gt(gt: torch.Tensor, n: int) -> torch.Tensor:
    """对 (B, M, 3) 的 GT 在第二维做无放回随机子采样到 n。
    与 FPS 相比的近似条件：GT 在预处理时已是均匀采样。"""
    B, M, _ = gt.shape
    if n == M:
        return gt
    if n > M:
        idx = torch.randint(0, M, (B, n), device=gt.device)
    else:
        idx = torch.stack([torch.randperm(M, device=gt.device)[:n] for _ in range(B)], dim=0)
    return torch.gather(gt, 1, idx.unsqueeze(-1).expand(-1, -1, 3))


# ── Dataset 包装：input / gt 不同点数（不修改 src/data/dataset.py） ─────
class CompletionDataset(SnowflakeDataset):
    """与 PCN/SnowflakeNet 原实验一致：input=2048, gt=16384"""

    def __init__(self, root_dir, split='train', input_points=2048, gt_points=16384):
        super().__init__(root_dir, split=split, num_points=input_points)
        self._gt_points = int(gt_points)
        self._input_points = int(input_points)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        input_pcd = np.load(os.path.join(self.input_path, file_name))
        gt_pcd = np.load(os.path.join(self.gt_path, file_name))
        input_pcd = self._resample(input_pcd, self._input_points)
        gt_pcd = self._resample(gt_pcd, self._gt_points)
        return torch.from_numpy(input_pcd).float(), torch.from_numpy(gt_pcd).float()


# ── 加载预训练权重（去 module. 前缀） ─────────────────────────────────
def load_pretrained(model: SnowflakeNet, ckpt_path: str):
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    state_dict = checkpoint['model']
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_sd[name] = v
    missing, unexpected = model.load_state_dict(new_sd, strict=True)
    assert not missing, f"Missing keys: {missing}"
    assert not unexpected, f"Unexpected keys: {unexpected}"
    log.info(f"预训练权重加载成功：{ckpt_path}")


# ── per-stage same-resolution Chamfer ──────────────────────────────────
def staged_loss(outputs, gt: torch.Tensor, weights):
    """对每个 stage 用相同点数的 GT 子采样后计算 CD。
    outputs: list[Tensor] (B, N_i, 3)。weights 长度需与 outputs 一致。"""
    losses = []
    for w, out in zip(weights, outputs):
        if w == 0.0:
            continue
        n = out.shape[1]
        gt_sub = random_subsample_gt(gt, n)
        losses.append(w * chamfer_l1(out, gt_sub))
    return sum(losses)


# ── 验证 ────────────────────────────────────────────────────────────────
@torch.no_grad()
def validate(model, loader, device, max_batches=None, log_wandb_3d=False, log_step=0):
    model.eval()
    total_cd = 0.0
    n_batch = 0
    for i, (inp, gt) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        inp, gt = inp.to(device, non_blocking=True), gt.to(device, non_blocking=True)
        outputs = model(inp)
        pred = outputs[-1]  # 与 GT 同分辨率（16384）
        if log_wandb_3d and i == 0:
            wandb_utils.log_val_pointclouds(
                gt[0].detach().cpu().numpy(),
                pred[0].detach().cpu().numpy(),
                step=log_step,
                prefix="val",
            )
        total_cd += chamfer_l1(pred, gt).item()
        n_batch += 1
    model.train()
    return total_cd / max(n_batch, 1)


def save_ckpt(state, path):
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)


# ── 主函数 ─────────────────────────────────────────────────────────────
def main(args):
    wandb_run = wandb_utils.init_wandb(args)
    try:
        _run_training(args, wandb_run)
    finally:
        wandb_utils.finish_wandb(wandb_run)


def _run_training(args, wandb_run):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"使用设备：{device}")

    train_ds = CompletionDataset(
        args.data_root, split='train',
        input_points=args.input_points, gt_points=args.gt_points,
    )
    val_ds = CompletionDataset(
        args.data_root, split='test',
        input_points=args.input_points, gt_points=args.gt_points,
    )
    log.info(f"训练集：{len(train_ds)} 样本  验证集：{len(val_ds)} 样本")
    log.info(f"input_points={args.input_points}  gt_points={args.gt_points}")
    if len(train_ds) >= 100_000:
        log.info(
            "数据集规模较大（如 130k 级）：DataLoader 为流式迭代，内存/显存主要由 batch 与模型决定；"
            "已启用 step 级 mini-val，无需整 epoch 才看到验证指标。"
        )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    model = SnowflakeNet(up_factors=[1, 4, 8]).to(device)
    load_pretrained(model, args.ckpt_pretrain)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    # 注意：scheduler 改为 step 级触发，patience 单位 = mini-val 次数
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5,
        patience=args.lr_patience, min_lr=1e-6,
    )

    os.makedirs(args.save_dir, exist_ok=True)
    best_cd = float('inf')
    best_path = os.path.join(args.save_dir, 'ckpt-best.pth')
    last_path = os.path.join(args.save_dir, 'ckpt-last.pth')

    weights = [float(x) for x in args.stage_weights.split(',')]
    assert len(weights) == 4, "--stage-weights 需要 4 个数, 对应 [Pc,P1,P2,P3]"

    global_step = 0
    sanity_printed = False

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_steps_in_epoch = 0

        for i, (inp, gt) in enumerate(train_loader):
            inp = inp.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(inp)

            if not sanity_printed:
                msg = "\n==============================\n【维度校验】Outputs shapes:\n"
                for k, o in enumerate(outputs):
                    msg += f"  Stage {k}: {tuple(o.shape)}\n"
                msg += f"  GT shape : {tuple(gt.shape)}\n"
                msg += f"  P3 == GT? {outputs[-1].shape[1] == gt.shape[1]}\n"
                msg += "==============================\n"
                log.info(msg)
                assert outputs[-1].shape[1] == gt.shape[1], (
                    f"P3({outputs[-1].shape[1]}) 与 GT({gt.shape[1]}) 仍不同分辨率，"
                    "请确认已用 --num-gt 16384 重新预处理。"
                )
                sanity_printed = True

            loss = staged_loss(outputs, gt, weights)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_steps_in_epoch += 1
            global_step += 1

            # ── step 级日志（文件 + W&B） ──
            if global_step % args.log_every == 0:
                lr_now = optimizer.param_groups[0]['lr']
                log.info(
                    f"Epoch {epoch:03d} | step {i+1:5d}/{len(train_loader)} "
                    f"| gstep {global_step:7d} | loss {loss.item():.6f} "
                    f"| lr {lr_now:.2e}"
                )
                wandb_utils.log_train_metrics(loss.item(), lr_now, global_step)

            # ── step 级 mini-val + 调度 + best/last 保存 ──
            if global_step % args.val_every_steps == 0:
                val_cd = validate(
                    model, val_loader, device,
                    max_batches=args.val_batches,
                    log_wandb_3d=(wandb_run is not None),
                    log_step=global_step,
                )
                scheduler.step(val_cd)
                lr_now = optimizer.param_groups[0]['lr']
                log.info(
                    f"[mini-val] gstep {global_step:7d} | val_cd {val_cd:.6f} "
                    f"| lr {lr_now:.2e} | best {best_cd:.6f}"
                )
                wandb_utils.log_wandb_scalars(
                    {
                        "val/mini_chamfer": val_cd,
                        "val/learning_rate": lr_now,
                        "val/best_chamfer": best_cd,
                    },
                    global_step,
                )
                save_ckpt(
                    {'epoch': epoch, 'global_step': global_step,
                     'model': model.state_dict(), 'val_cd': val_cd},
                    last_path,
                )
                if val_cd < best_cd:
                    best_cd = val_cd
                    save_ckpt(
                        {'epoch': epoch, 'global_step': global_step,
                         'model': model.state_dict(), 'val_cd': val_cd},
                        best_path,
                    )
                    log.info(f"  ✅ 新最优 val_cd={best_cd:.6f}，已保存至 {best_path}")

            # ── 周期性常规快照 ──
            if args.save_every_steps > 0 and global_step % args.save_every_steps == 0:
                snap = os.path.join(args.save_dir, f'ckpt-step{global_step:08d}.pth')
                save_ckpt(
                    {'epoch': epoch, 'global_step': global_step,
                     'model': model.state_dict()},
                    snap,
                )

        avg = running_loss / max(n_steps_in_epoch, 1)
        # epoch 末做一次完整 val（不参与 scheduler，仅记录）
        full_val = validate(model, val_loader, device, max_batches=None)
        log.info(
            f"=== Epoch {epoch:03d}/{args.epochs} done | train_loss_avg {avg:.6f} "
            f"| full_val_cd {full_val:.6f} | best {best_cd:.6f} ==="
        )
        wandb_utils.log_wandb_scalars(
            {
                "epoch": epoch,
                "train/epoch_avg_loss": avg,
                "val/full_chamfer": full_val,
                "val/best_chamfer": best_cd,
            },
            global_step,
        )

    log.info(f"训练完成。最优 val_cd = {best_cd:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default='/home/csy/SnowflakeNet_FPFH_ICP/data/processed/modelnet40',
                        help='processed/<dataset> 路径，下含 train/{input,gt} 与 test/{input,gt}')
    parser.add_argument('--ckpt_pretrain', type=str,
                        default='/home/csy/SnowflakeNet_FPFH_ICP/Snet/SnowflakeNet-main/'
                                'completion/checkpoints/ckpt-best-pcn-cd_l1.pth')
    parser.add_argument('--save_dir', type=str,
                        default='/home/csy/SnowflakeNet_FPFH_ICP/checkpoints/snet_finetune')

    # 与原 PCN/SnowflakeNet 一致
    parser.add_argument('--input_points', type=int, default=2048)
    parser.add_argument('--gt_points', type=int, default=16384)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)

    # 各阶段权重 [Pc, P1, P2, P3]
    parser.add_argument('--stage-weights', type=str, default='1.0,1.0,1.0,1.0',
                        help='对应 Pc(256), P1(512), P2(2048), P3(gt_points) 的 CD 权重')

    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='禁用 Weights & Biases（仍写文件日志与本地 checkpoint）',
    )

    # step 级触发
    parser.add_argument('--log-every', type=int, default=20,
                        help='每 N 个 global step 写一次 train 日志与 W&B')
    parser.add_argument('--val-every-steps', type=int, default=2000)
    parser.add_argument('--val-batches', type=int, default=200,
                        help='mini-val 时只取前 N 个 batch（不影响 epoch 末 full-val）')
    parser.add_argument('--save-every-steps', type=int, default=10000,
                        help='周期性快照间隔；<=0 关闭')
    parser.add_argument('--lr-patience', type=int, default=5,
                        help='ReduceLROnPlateau 单位是 mini-val 次数')

    args = parser.parse_args()
    main(args)
