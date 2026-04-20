"""Shared Weights & Biases helpers for finetuning scripts."""

from __future__ import annotations

import logging
from argparse import Namespace
from typing import Any, Optional

log = logging.getLogger(__name__)


def namespace_to_wandb_config(ns: Namespace) -> dict[str, Any]:
    """Serialize argparse Namespace for wandb.config (JSON-friendly)."""
    out: dict[str, Any] = {}
    for k, v in vars(ns).items():
        if v is None or isinstance(v, (bool, int, float, str)):
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        else:
            out[k] = str(v)
    return out


def init_wandb(
    args: Namespace,
    *,
    project: str = "SnowflakeNet_Finetune",
    job_type: str = "train",
):
    """Initialize wandb.run or return None if disabled / not installed."""
    if getattr(args, "no_wandb", False):
        log.info("W&B disabled (--no-wandb).")
        return None
    try:
        import wandb
    except ImportError:
        log.warning("wandb is not installed; `pip install wandb` to enable W&B.")
        return None
    cfg = namespace_to_wandb_config(args)
    run = wandb.init(project=project, config=cfg, job_type=job_type)
    return run


def log_wandb_scalars(metrics: dict, step: int) -> None:
    try:
        import wandb

        if wandb.run is None:
            return
    except ImportError:
        return
    wandb.log(metrics, step=step)


def finish_wandb(run) -> None:
    if run is None:
        return
    try:
        import wandb

        wandb.finish()
    except Exception as e:
        log.warning("wandb.finish() failed: %s", e)


def log_train_metrics(loss: float, learning_rate: float, global_step: int) -> None:
    try:
        import wandb

        if wandb.run is None:
            return
    except ImportError:
        return
    wandb.log(
        {
            "train/loss": loss,
            "train/learning_rate": learning_rate,
            "train/global_step": global_step,
        },
        step=global_step,
    )


def log_val_pointclouds(
    gt_xyz,
    pred_xyz,
    *,
    step: int,
    prefix: str = "val",
) -> None:
    """Log GT and prediction as wandb.Object3D (numpy (N,3))."""
    try:
        import numpy as np
        import wandb

        if wandb.run is None:
            return
    except ImportError:
        return
    gt_xyz = np.asarray(gt_xyz, dtype=np.float32)
    pred_xyz = np.asarray(pred_xyz, dtype=np.float32)
    wandb.log(
        {
            f"{prefix}/gt_pointcloud": wandb.Object3D(gt_xyz),
            f"{prefix}/pred_pointcloud": wandb.Object3D(pred_xyz),
        },
        step=step,
    )
