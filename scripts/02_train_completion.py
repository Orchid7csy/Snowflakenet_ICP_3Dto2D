"""
scripts/02_train_completion.py

SnowflakeNet 微调脚本（与原 PCN/SnowflakeNet 实验对齐版本）

关键修复（相对旧版）：
  1) GT=16384 与 up_factors=[1,4,8] 同分辨率（P3 也是 16384），消除“稀疏 pred ↔ 密集 GT”
     的 Chamfer 跨分辨率退化。Pc/P1/P2 用 farthest point sampling 将 GT 降到与各 stage 相同
     的点数后再算 CD（CPU 回退为随机子采样）。
  2) Step 级验证 / 调度 / 保存（默认每 2000 step 一次 mini-val），Epoch 级仅用作汇总日志。
  3) 维度自检：第一次 forward 打印各 stage 形状，确认 P3 == GT_points。

注意：
  - 必须重新预处理（gt 目录里的 .npy 必须是 16384 点）。运行
        python scripts/00_preprocess_pcn.py
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
log_dir = os.path.join(project_root, 'checkpoints', 'snet_finetune')
os.makedirs(log_dir, exist_ok=True)
for p in [project_root, snet_root]:
    if p not in sys.path:
        sys.path.insert(0, p)

_scripts_dir = os.path.dirname(os.path.abspath(__file__))
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)
import wandb_utils  # noqa: E402

from src.models.chamfer import chamfer_l1_symmetric  # noqa: E402
from src.data.pcn_dataset import PCNRotAugCompletionDataset, CompletionDataset  # noqa: E402
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


# ── Chamfer Distance L1（与 Snet test / 表 1 一致：(d1̄+d2̄)/2，见 snet_pcn_chamfer） ─
_chamfer_fn = chamfer_3DDist()


def random_subsample_gt(gt: torch.Tensor, n: int) -> torch.Tensor:
    """对 (B, M, 3) 的 GT 在第二维做随机子采样到 n（CUDA 不可用时 FPS 的回退）。"""
    B, M, _ = gt.shape
    if n == M:
        return gt
    if n > M:
        idx = torch.randint(0, M, (B, n), device=gt.device)
    else:
        idx = torch.stack([torch.randperm(M, device=gt.device)[:n] for _ in range(B)], dim=0)
    return torch.gather(gt, 1, idx.unsqueeze(-1).expand(-1, -1, 3))


def fps_subsample_gt(gt: torch.Tensor, n: int) -> torch.Tensor:
    """Farthest point sampling (B, M, 3) -> (B, n, 3)。CUDA 上复用 pointnet2_ops；CPU 回退随机。"""
    _, M, _ = gt.shape
    if n >= M:
        return gt
    if not gt.is_cuda:
        return random_subsample_gt(gt, n)
    try:
        from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation
    except ImportError:
        return random_subsample_gt(gt, n)
    idx = furthest_point_sample(gt.contiguous(), int(n))
    feats = gt.transpose(1, 2).contiguous()
    gathered = gather_operation(feats, idx)
    return gathered.transpose(1, 2).contiguous()


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


# ── per-stage same-resolution Chamfer + Preservation (+ optional symmetry) ─
def staged_loss(
    outputs,
    gt: torch.Tensor,
    inp: torch.Tensor,
    weights,
    lambda_pres: float = 0.0,
    *,
    lambda_sym: float = 0.0,
):
    """
    返回 (total_loss, loss_completion, loss_preservation, loss_symmetry)。

    - loss_completion: 各 stage (Pc/P1/P2/P3) 与同分辨率 GT 的**对称** L1 CD 加权和。
    - loss_preservation: SnowflakeNet §3.4 的 **保真（partial matching）损失**——官方实现
      (``completion/utils/loss_util.py::Completionloss.chamfer_partial_l1``) 调用
      ``chamfer_dist(partial, P3)`` 并对 **第一方向距离** 取 ``mean(sqrt(dist1))``，即
      「对每个残缺输入点，找其在最终输出 P3 中的最近点」。这是**单向**约束：强制 P3
      覆盖所有已观测点，避免模型回归到数据集平均形状。**不要反向**（那会把 P3 拉向
      稀疏输入）。官方总 loss 中该项权重为 1.0，这里通过 ``lambda_pres`` 可配。
    - loss_symmetry（可选）: 对 P3 沿 z（PCA 主轴）镜像后的对称 CD，``lambda_sym`` 默认 0 关闭。
    """
    losses = []
    for w, out in zip(weights, outputs):
        if w == 0.0:
            continue
        n = out.shape[1]
        gt_sub = fps_subsample_gt(gt, n)
        losses.append(w * chamfer_l1_symmetric(out, gt_sub))
    loss_completion = sum(losses) if losses else torch.zeros((), device=gt.device)

    # 保真项：对 (inp, P3) 求 chamfer，取 dist1（每个 inp 点到最近 P3 的距离）
    # 的 sqrt().mean()，与官方 chamfer_partial_l1(partial, P3) 完全一致。
    p3 = outputs[-1]
    dist1, _dist2, _, _ = _chamfer_fn(inp, p3)
    loss_preservation = (dist1 + 1e-8).sqrt().mean()

    if float(lambda_sym) > 0.0:
        p3_flip = p3.clone()
        p3_flip[..., 2] *= -1.0
        loss_symmetry = chamfer_l1_symmetric(p3, p3_flip)
    else:
        loss_symmetry = torch.zeros((), device=gt.device)

    total = (
        loss_completion
        + float(lambda_pres) * loss_preservation
        + float(lambda_sym) * loss_symmetry
    )
    return total, loss_completion, loss_preservation, loss_symmetry


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
        total_cd += chamfer_l1_symmetric(pred, gt).item()
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


def _train_input_dir(root: str) -> str:
    return os.path.join(root, "train", "input")


def _total_optimizer_steps(args, len_easy: int, len_hard: int) -> int:
    """与下方 epoch/current_loader 循环一致的训练步总数（用于 Cosine T_max）。"""
    total = 0
    for epoch in range(1, args.epochs + 1):
        if epoch <= args.epoch_hard_start:
            total += len_easy
        else:
            total += len_hard
    return total


def _run_training(args, wandb_run):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log.info(f"使用设备：{device}")

    # ── 课程式学习：两个训练集（easy/hard），验证始终使用 hard 的 test 划分 ──
    # ModelNet: easy=仅 HPR；hard=HPR+块丢弃+噪声（legacy 预处理）。
    # PCN: 00_preprocess_pcn.py --mode easy 产出每模型 HPR 最佳视角 easy 集；
    # --mode hard/默认产出所有 PCN partial 视角 hard 集。无单独 easy 目录时回退到 hard。
    data_root_hard = args.data_root_hard or args.data_root
    data_root_easy = args.data_root_easy or data_root_hard
    if not os.path.isdir(_train_input_dir(data_root_easy)):
        if os.path.abspath(data_root_easy) != os.path.abspath(data_root_hard):
            log.warning(
                "data-root-easy 无 train/input（%s），回退为 data-root-hard，课程学习关闭。",
                _train_input_dir(data_root_easy),
            )
        data_root_easy = data_root_hard
    # 当 easy/hard 根目录不同且 epoch_hard_start < epochs 时，才真正发生切换。
    use_curriculum = (data_root_easy != data_root_hard) and (args.epoch_hard_start >= 1)

    def _make_ds(root, split):
        if args.use_pcn_rot_aug:
            return PCNRotAugCompletionDataset(
                root, split=split,
                input_points=args.input_points, gt_points=args.gt_points,
                rot_aug=True, rot_mode=args.pcn_rot_mode,
            )
        return CompletionDataset(
            root, split=split,
            input_points=args.input_points, gt_points=args.gt_points,
        )

    train_ds_hard = _make_ds(data_root_hard, 'train')
    val_ds = _make_ds(data_root_hard, 'test')
    train_loader_hard = DataLoader(
        train_ds_hard, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    if use_curriculum:
        train_ds_easy = _make_ds(data_root_easy, 'train')
        train_loader_easy = DataLoader(
            train_ds_easy, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True, drop_last=True,
        )
        log.info(
            f"课程式学习启用: easy={data_root_easy} ({len(train_ds_easy)} 样本) "
            f"-> hard={data_root_hard} ({len(train_ds_hard)} 样本) @ epoch {args.epoch_hard_start}"
        )
    else:
        train_loader_easy = train_loader_hard
        log.info(f"课程式学习未启用: 始终使用 {data_root_hard} ({len(train_ds_hard)} 样本)")

    log.info(f"验证集 (hard/test): {len(val_ds)} 样本")
    log.info(f"input_points={args.input_points}  gt_points={args.gt_points}  λ_pres={args.lambda_pres}")

    model = SnowflakeNet(up_factors=[1, 4, 8]).to(device)
    load_pretrained(model, args.ckpt_pretrain)

    len_easy = len(train_loader_easy)
    len_hard = len(train_loader_hard)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler_cosine = None
    scheduler_plateau = None
    if args.scheduler == "cosine":
        total_opt_steps = _total_optimizer_steps(args, len_easy, len_hard)
        scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_opt_steps, eta_min=args.eta_min,
        )
        log.info(
            f"学习率调度: CosineAnnealingLR(T_max={total_opt_steps}, eta_min={args.eta_min:g}) "
            f"（easy steps/epoch={len_easy}, hard={len_hard}）"
        )
    elif args.scheduler == "plateau":
        scheduler_plateau = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5,
            patience=args.lr_patience, min_lr=args.eta_min,
        )
        log.info(
            f"学习率调度: ReduceLROnPlateau(patience={args.lr_patience} 次 mini-val, "
            f"factor=0.5, min_lr={args.eta_min:g})"
        )
    else:
        raise ValueError(f"未知 --scheduler: {args.scheduler}")

    if not args.skip_step0_sanity:
        step0_cd = validate(model, val_loader, device, max_batches=None)
        lr0 = optimizer.param_groups[0]['lr']
        log.info(
            f"[sanity] 训练前 val CD（全 test，无权重更新）: {step0_cd:.6f}  |  lr {lr0:.2e}  "
            f"（PCN 预训练 paper 约 0.007；若 >>0.02 请检查预处理与数据分布）"
        )
        wandb_utils.log_wandb_scalars({"val/step0_chamfer": step0_cd, "val/step0_lr": lr0}, 0)

    os.makedirs(args.save_dir, exist_ok=True)
    best_cd = float('inf')
    best_path = os.path.join(args.save_dir, 'ckpt-best.pth')
    last_path = os.path.join(args.save_dir, 'ckpt-last.pth')

    weights = [float(x) for x in args.stage_weights.split(',')]
    assert len(weights) == 4, "--stage-weights 需要 4 个数, 对应 [Pc,P1,P2,P3]"

    global_step = 0
    sanity_printed = False

    for epoch in range(1, args.epochs + 1):
        # 课程式调度：epoch <= epoch_hard_start 用 easy，其后切 hard。
        # 默认 epoch_hard_start=50 → 第 1..50 epoch 暖启动于 easy，第 51 epoch 起切 hard。
        if epoch <= args.epoch_hard_start:
            current_loader = train_loader_easy
            stage_tag = "easy"
        else:
            current_loader = train_loader_hard
            stage_tag = "hard"
            if use_curriculum and epoch == args.epoch_hard_start + 1:
                log.info(
                    f"Epoch {epoch}: 启动课程式学习高难度阶段，切换至 Hard DataLoader "
                    f"({data_root_hard})"
                )

        model.train()
        running_loss = 0.0
        running_loss_comp = 0.0
        running_loss_pres = 0.0
        running_loss_sym = 0.0
        n_steps_in_epoch = 0

        for i, (inp, gt) in enumerate(current_loader):
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

            loss, loss_comp, loss_pres, loss_sym = staged_loss(
                outputs, gt, inp, weights,
                lambda_pres=args.lambda_pres,
                lambda_sym=args.lambda_sym,
            )
            loss.backward()
            optimizer.step()
            if scheduler_cosine is not None:
                scheduler_cosine.step()

            running_loss += loss.item()
            running_loss_comp += loss_comp.item()
            running_loss_pres += loss_pres.item()
            running_loss_sym += loss_sym.item()
            n_steps_in_epoch += 1
            global_step += 1

            # ── step 级日志（文件 + W&B） ──
            if global_step % args.log_every == 0:
                lr_now = optimizer.param_groups[0]['lr']
                log.info(
                    f"Epoch {epoch:03d} [{stage_tag}] | step {i+1:5d}/{len(current_loader)} "
                    f"| gstep {global_step:7d} | loss {loss.item():.6f} "
                    f"(comp {loss_comp.item():.6f}, pres {loss_pres.item():.6f}) "
                    f"| lr {lr_now:.2e}"
                )
                wandb_utils.log_train_metrics(loss.item(), lr_now, global_step)
                scalars = {
                    "train/loss_total": loss.item(),
                    "train/loss_completion": loss_comp.item(),
                    "train/loss_preservation": loss_pres.item(),
                    "train/curriculum_stage": 0 if stage_tag == "easy" else 1,
                }
                if args.lambda_sym > 0:
                    scalars["train/loss_symmetry"] = loss_sym.item()
                wandb_utils.log_wandb_scalars(scalars, global_step)

            # ── step 级 mini-val + 调度 + best/last 保存 ──
            if global_step % args.val_every_steps == 0:
                val_cd = validate(
                    model, val_loader, device,
                    max_batches=args.val_batches,
                    log_wandb_3d=(wandb_run is not None),
                    log_step=global_step,
                )
                if scheduler_plateau is not None:
                    scheduler_plateau.step(val_cd)
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

        n = max(n_steps_in_epoch, 1)
        avg = running_loss / n
        avg_comp = running_loss_comp / n
        avg_pres = running_loss_pres / n
        avg_sym = running_loss_sym / n
        full_val = validate(model, val_loader, device, max_batches=None)
        sym_suffix = f", sym {avg_sym:.6f}" if args.lambda_sym > 0 else ""
        log.info(
            f"=== Epoch {epoch:03d}/{args.epochs} [{stage_tag}] done | "
            f"loss_avg {avg:.6f} (comp {avg_comp:.6f}, pres {avg_pres:.6f}{sym_suffix}) "
            f"| full_val_cd {full_val:.6f} | best {best_cd:.6f} ==="
        )
        wb_epoch = {
            "epoch": epoch,
            "train/epoch_avg_loss": avg,
            "train/epoch_avg_loss_completion": avg_comp,
            "train/epoch_avg_loss_preservation": avg_pres,
            "val/full_chamfer": full_val,
            "val/best_chamfer": best_cd,
        }
        if args.lambda_sym > 0:
            wb_epoch["train/epoch_avg_loss_symmetry"] = avg_sym
        wandb_utils.log_wandb_scalars(wb_epoch, global_step)

    log.info(f"训练完成。最优 val_cd = {best_cd:.6f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                        default=os.path.join(project_root, 'data', 'processed', 'PCN_far_cano_in2048_gt16384'),
                        help='processed/<dataset> 路径，下含 train/{input,gt} 与 test/{input,gt}')
    parser.add_argument('--ckpt_pretrain', type=str,
                        default=os.path.join(
                            project_root, 'Snet', 'SnowflakeNet-main', 'completion',
                            'checkpoints', 'ckpt-best-pcn-cd_l1.pth',
                        ))
    parser.add_argument('--save_dir', type=str,
                        default=os.path.join(project_root, 'checkpoints', 'snet_finetune'))

    # 与原 PCN/SnowflakeNet 一致
    parser.add_argument('--input_points', type=int, default=2048)
    parser.add_argument('--gt_points', type=int, default=16384)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='微调学习率；与原 SnowflakeNet PCN 训练 1e-4 对齐（旧默认 1e-5 过小易停滞）',
    )
    parser.add_argument(
        '--scheduler', type=str, default='cosine', choices=('cosine', 'plateau'),
        help='cosine=每 optimizer step 的余弦退火；plateau=mini-val 触发 ReduceLROnPlateau',
    )
    parser.add_argument(
        '--eta-min', type=float, default=1e-6,
        help='Cosine/Plateau 的最小学习率',
    )

    parser.add_argument(
        '--use-pcn-rot-aug', action='store_true',
        help='训练集对 input/GT 同步旋转增强（默认关闭；PCA 对齐下建议 signflip 而非 so3）',
    )
    parser.add_argument(
        '--pcn-rot-mode', type=str, default='signflip', choices=('so3', 'yaw', 'signflip'),
        help='so3=全向；yaw=绕 Z；signflip=主轴符号翻转（覆盖 PCA 二义性，默认）',
    )

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
    parser.add_argument('--lr-patience', type=int, default=3,
                        help='ReduceLROnPlateau：单位是 mini-val 次数')

    # ── 保真损失 (SnowflakeNet §3.4) ──
    parser.add_argument(
        '--lambda-pres', type=float, default=1.0,
        help='保真损失权重（与官方 completion loss 中 partial 项系数 1.0 一致）',
    )
    parser.add_argument(
        '--lambda-sym', type=float, default=0.0,
        help='可选：P3 沿 z 镜像的对称 CD 权重；默认 0 关闭',
    )
    parser.add_argument(
        '--skip-step0-sanity', action='store_true',
        help='跳过训练前全量 val 健全性检查（val/step0_chamfer）',
    )

    # ── 课程式学习 ──
    parser.add_argument(
        '--epoch-hard-start', type=int, default=50,
        help='前 N 个 epoch (epoch <= N) 使用 easy DataLoader 暖启动 SPD 粗->细展开，'
             '之后 (epoch > N) 切换至 hard。默认 50：第 51 epoch 起启动高难度阶段。',
    )
    parser.add_argument(
        '--data-root-easy', type=str,
        default=os.path.join(project_root, 'data', 'processed', 'PCN_hpr_easy_cano_in2048_gt16384'),
        help='课程 easy：须含 train/input/*.npy。ModelNet: 01 --mode easy；'
             'PCN: 先运行 01_compute_best_view.py，再运行 00_preprocess_pcn.py --mode easy；'
             '若目录不存在则自动回退到 hard。',
    )
    parser.add_argument(
        '--data-root-hard', type=str,
        default=os.path.join(project_root, 'data', 'processed', 'PCN_far_cano_in2048_gt16384'),
        help='课程 hard / 验证集：ModelNet 用 01 --mode hard；'
             'PCN 用 preprocess_pcn_bbox_pca。验证始终用本目录下 test。',
    )

    args = parser.parse_args()
    main(args)
