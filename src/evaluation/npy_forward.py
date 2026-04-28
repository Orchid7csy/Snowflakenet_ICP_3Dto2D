"""预处理 .npy 上 SnowflakeNet 前向（与 test_pcn_processed_cdl1 一致）。"""
from __future__ import annotations

import numpy as np
import torch

from src.evaluation.upsample_snet import UpSamplePointsSnet
from src.pose_estimation.postprocess import FilterConfig, filter_completion_spurious


def resample_points(data: np.ndarray, target_n: int = 2048) -> np.ndarray:
    if data.shape[0] < target_n:
        idx = np.random.choice(data.shape[0], target_n, replace=True)
        return data[idx]
    if data.shape[0] > target_n:
        idx = np.random.choice(data.shape[0], target_n, replace=False)
        return data[idx]
    return data


def forward_p3(
    model: torch.nn.Module,
    part_xyz: np.ndarray,
    device: torch.device,
    do_comp_filter: bool,
    *,
    legacy_partial: bool,
    n_input_points: int = 2048,
) -> np.ndarray:
    if legacy_partial:
        part_rs = resample_points(part_xyz, n_input_points)
    else:
        up = UpSamplePointsSnet({"n_points": n_input_points})
        part_rs = up(part_xyz.astype(np.float32))
    with torch.no_grad():
        t = torch.from_numpy(part_rs).float().unsqueeze(0).to(device)
        p3 = model(t)[-1].squeeze(0).detach().cpu().numpy()
    if do_comp_filter:
        cfg = FilterConfig()
        p3, _ = filter_completion_spurious(p3, part_rs, cfg=cfg)
    return p3


def forward_from_npy(
    model: torch.nn.Module,
    part_xyz: np.ndarray,
    device: torch.device,
    *,
    input_mode: str,
    do_comp_filter: bool,
    n_input_points: int,
) -> np.ndarray:
    if input_mode == "direct":
        p = part_xyz.astype(np.float32)
        if p.shape[0] != n_input_points:
            p = resample_points(p, n_input_points)
        with torch.no_grad():
            t = torch.from_numpy(p).float().unsqueeze(0).to(device)
            p3 = model(t)[-1].squeeze(0).detach().cpu().numpy()
        if do_comp_filter:
            cfg = FilterConfig()
            p3, _ = filter_completion_spurious(p3, p, cfg=cfg)
        return p3
    legacy = input_mode == "legacy"
    return forward_p3(
        model,
        part_xyz.astype(np.float32),
        device,
        do_comp_filter=do_comp_filter,
        legacy_partial=legacy,
        n_input_points=n_input_points,
    )
