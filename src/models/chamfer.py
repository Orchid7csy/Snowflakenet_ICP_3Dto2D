"""
与 Snet Completionloss.chamfer_l1 / test.py 一致的对称 L1 Chamfer: (D_pred_to_gt + D_gt_to_pred) / 2。

需在使用前将项目根与 Snet 根目录加入 sys.path（与 02_train_completion / batch_infer 相同）。

Snet 的 chamfer_3D CUDA 扩展在 CPU 张量上不可用；此时使用下方纯 PyTorch 实现，与扩展在
数学上对齐（成对欧氏距离平方 + sqrt + 双向 mean），大点云上较慢，训练/推理仍应使用 GPU。
"""

from __future__ import annotations

import torch

_chamfer_module = None


def _get_chamfer():
    """仅在需要 CUDA 路径时加载 Snet 扩展，CPU 单测可不触达。"""
    global _chamfer_module
    if _chamfer_module is None:
        from loss_functions.Chamfer3D.dist_chamfer_3D import chamfer_3DDist

        _chamfer_module = chamfer_3DDist()
    return _chamfer_module


def _chamfer_l1_symmetric_cpu(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    纯 torch，pred/gt: (B, N, 3) / (B, M, 3)，在 CPU 或小 batch 上可用；显存为 O(B*N*M)。
    """
    eps = 1e-8
    d2 = (pred[:, :, None, :] - gt[:, None, :, :]).pow(2).sum(-1)  # (B, N, M)
    d1, _ = d2.min(dim=2)  # (B, N) pred 每点到最近 gt
    d2m, _ = d2.min(dim=1)  # (B, M) gt 每点到最近 pred
    s1 = (d1 + eps).sqrt().mean(dim=1)
    s2 = (d2m + eps).sqrt().mean(dim=1)
    return 0.5 * (s1 + s2).mean()


def chamfer_l1_symmetric(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """
    pred: (B, N, 3), gt: (B, M, 3)。
    对 batch 内每个样本各算 (mean sqrt(d1) + mean sqrt(d2)) / 2，再对 batch 取平均。
    """
    if pred.is_cuda and gt.is_cuda and pred.get_device() == gt.get_device():
        dist1, dist2, _, _ = _get_chamfer()(pred, gt)
        eps = 1e-8
        d1 = (dist1 + eps).sqrt().mean(dim=1)
        d2 = (dist2 + eps).sqrt().mean(dim=1)
        return 0.5 * (d1 + d2).mean()
    if pred.is_cuda or gt.is_cuda:
        raise ValueError("pred 与 gt 需在同一设备且同为 CPU 或同为 CUDA，当前混合/不一致")
    return _chamfer_l1_symmetric_cpu(pred, gt)
