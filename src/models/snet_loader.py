"""SnowflakeNet 加载与单前向补全。"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import torch

from models.model_completion import SnowflakeNet


def load_snowflakenet(ckpt_path: str) -> torch.nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_loc = device if device.type == "cuda" else "cpu"
    model = SnowflakeNet(up_factors=[1, 4, 8])
    checkpoint = torch.load(ckpt_path, map_location=map_loc)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def complete_points(model: torch.nn.Module, input_points: np.ndarray) -> np.ndarray:
    device = next(model.parameters()).device
    pts = input_points.astype(np.float32)
    if pts.shape[0] < 2048:
        idx = np.random.choice(pts.shape[0], 2048, replace=True)
        pts = pts[idx]
    elif pts.shape[0] > 2048:
        idx = np.random.choice(pts.shape[0], 2048, replace=False)
        pts = pts[idx]
    x = torch.from_numpy(pts).unsqueeze(0).to(device)
    outs = model(x)
    dense = outs[-1].squeeze(0).detach().cpu().numpy().astype(np.float32)
    return dense
