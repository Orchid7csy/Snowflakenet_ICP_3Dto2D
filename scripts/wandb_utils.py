"""Shared Weights & Biases helpers for finetuning scripts."""

from __future__ import annotations

import logging
import os
import sys
from argparse import Namespace
from typing import Any, Optional

log = logging.getLogger(__name__)


def _wandb_version_triplet() -> tuple[int, int, int]:
    """Parse wandb.__version__ into (major, minor, patch) for comparisons."""
    try:
        import re

        import wandb

        raw = getattr(wandb, "__version__", "0.0.0") or "0.0.0"
        m = re.match(r"(\d+)\.(\d+)\.(\d+)", raw.strip())
        if m:
            return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except Exception:
        pass
    return (0, 0, 0)


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

    # New wandb_v1_* keys need wandb>=0.22.3, which requires Python>=3.8 (pip cannot install on 3.7).
    key = str(os.environ.get("WANDB_API_KEY", "") or "").strip()
    wv = _wandb_version_triplet()
    if key.startswith("wandb_v1_"):
        if sys.version_info < (3, 8):
            log.error(
                "WANDB_API_KEY is wandb_v1 format but Python %s cannot install wandb>=0.22.3. "
                "Options: (1) conda env Python>=3.10: see environment_wandb_online.yml "
                "(2) train with --no-wandb.",
                sys.version.split()[0],
            )
            return None
        if wv < (0, 22, 3):
            log.error(
                "wandb %s cannot use wandb_v1 API keys (need >=0.22.3). "
                "Upgrade: pip install -U 'wandb>=0.22.3' using Python>=3.8, "
                "or use environment_wandb_online.yml / conda create -n s310 python=3.10.",
                getattr(wandb, "__version__", "?"),
            )
            return None

    cfg = namespace_to_wandb_config(args)
    try:
        return wandb.init(project=project, config=cfg, job_type=job_type)
    except ValueError as e:
        err = str(e)
        if "40 characters" in err or "API key" in err.lower():
            log.error(
                "W&B API key was rejected. New keys (`wandb_v1_*`) require wandb>=0.22.3 "
                "(Python>=3.8). On Python 3.7 only wandb<=0.18 is installable — use conda env "
                "`environment_wandb_online.yml`, or train with --no-wandb.\n"
                "    conda env create -f environment_wandb_online.yml\n"
                "    conda activate snowflake-wandb\n"
                "    pip install -U 'wandb>=0.22.3'"
            )
        raise


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
