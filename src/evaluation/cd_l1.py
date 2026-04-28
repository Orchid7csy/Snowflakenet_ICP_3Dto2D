"""PCN CD-L1×10³ 与论文表 1 常数。"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.models.chamfer import chamfer_l1_symmetric
from src.data.naming import PCN_TAXONOMY

TAXONOMY_TABLE1: list[tuple[str, str]] = [
    ("02691156", "Plane"),
    ("02933112", "Cabinet"),
    ("02958343", "Car"),
    ("03001627", "Chair"),
    ("03636649", "Lamp"),
    ("04256520", "Couch"),
    ("04379243", "Table"),
    ("04530566", "Boat"),
]

PAPER_SNOWFLAKE: dict[str, float] = {
    "Average": 7.21,
    "Plane": 4.29,
    "Cabinet": 9.16,
    "Car": 8.08,
    "Chair": 7.89,
    "Lamp": 6.07,
    "Couch": 9.23,
    "Table": 6.55,
    "Boat": 6.40,
}

_SLUG_TO_EN: Dict[str, str] = {t[2]: t[1] for t in PCN_TAXONOMY}
_STEM_CLASS_RE = re.compile(
    r"^pcn__[^_]+__(?P<slug>[\w]+)_(?P<synset>\d{8})__.+__view_\d+$"
)


def english_name_from_stem(stem: str) -> Optional[str]:
    m = _STEM_CLASS_RE.match(stem)
    if not m:
        return None
    slug = m.group("slug")
    return _SLUG_TO_EN.get(slug)


def cdl1_times_1e3(
    pred_xyz: np.ndarray, gt_xyz: np.ndarray, device: torch.device
) -> float:
    p = torch.from_numpy(pred_xyz.astype(np.float32)).float().unsqueeze(0).to(device)
    g = torch.from_numpy(gt_xyz.astype(np.float32)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        v = chamfer_l1_symmetric(p, g)
    return (v * 1e3).item()


def list_npy_pairs(processed_root: str, split: str) -> List[Tuple[str, str, str]]:
    import os

    inp_dir = os.path.join(processed_root, split, "input")
    gt_dir = os.path.join(processed_root, split, "gt")
    if not os.path.isdir(inp_dir) or not os.path.isdir(gt_dir):
        return []
    out: List[Tuple[str, str, str]] = []
    for name in sorted(os.listdir(inp_dir)):
        if not name.endswith(".npy"):
            continue
        stem = name[:-4]
        ip = os.path.join(inp_dir, name)
        gp = os.path.join(gt_dir, name)
        if not os.path.isfile(gp):
            continue
        out.append((stem, ip, gp))
    return out


def select_per_class(
    pairs: List[Tuple[str, str, str]], per_class: int, seed: int
) -> List[Tuple[str, str, str]]:
    import random

    if per_class <= 0:
        return pairs
    rng = random.Random(seed)
    by_class: Dict[str, List[Tuple[str, str, str]]] = defaultdict(list)
    for stem, ip, gp in pairs:
        cname = english_name_from_stem(stem) or "__unknown__"
        by_class[cname].append((stem, ip, gp))
    chosen: List[Tuple[str, str, str]] = []
    for cname in sorted(by_class.keys()):
        items = by_class[cname][:]
        rng.shuffle(items)
        chosen.extend(items[:per_class])
    return chosen
