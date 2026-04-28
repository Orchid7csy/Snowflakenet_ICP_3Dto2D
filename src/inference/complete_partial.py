"""单样本 partial 点云 → SnowflakeNet 补全（可保存/可视化）。"""
from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import open3d as o3d
import torch

from models.model_completion import SnowflakeNet
from src.pose_estimation.postprocess import FilterConfig, filter_completion_spurious


def _resample_points(data: np.ndarray, target_n: int = 2048) -> np.ndarray:
    if data.shape[0] < target_n:
        idx = np.random.choice(data.shape[0], target_n, replace=True)
        return data[idx]
    if data.shape[0] > target_n:
        idx = np.random.choice(data.shape[0], target_n, replace=False)
        return data[idx]
    return data


def _export_stage_outputs(stage_outputs, export_dir: str, stem: str) -> None:
    os.makedirs(export_dir, exist_ok=True)
    stage_names = ["pc_seed", "p1", "p2", "p3"]
    for name, pts in zip(stage_names, stage_outputs):
        out_path = os.path.join(export_dir, f"{stem}_{name}.npy")
        np.save(out_path, pts.astype(np.float32))
        print(f"[stage export] {name}: {out_path}")


def complete_partial_points(
    data: np.ndarray,
    weight_path: str,
    out_npy_path: str,
    show_vis: bool = True,
    export_stages_dir: str | None = None,
    stage_stem: str = "input",
    *,
    do_comp_filter: bool = True,
    filter_cfg: FilterConfig | None = None,
    save_unfiltered: bool = False,
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    map_loc = device if device.type == "cuda" else "cpu"

    model = SnowflakeNet(up_factors=[1, 4, 8])
    checkpoint = torch.load(weight_path, map_location=map_loc)
    state_dict = checkpoint["model"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    data = np.asarray(data, dtype=np.float32)
    data = _resample_points(data, target_n=2048)
    input_tensor = torch.from_numpy(data).unsqueeze(0).to(device)

    with torch.no_grad():
        ret = model(input_tensor)
        stage_outputs = [o.squeeze(0).detach().cpu().numpy() for o in ret]
        dense_points = stage_outputs[-1].copy()

    unf = dense_points.copy() if (save_unfiltered and do_comp_filter) else None
    if do_comp_filter:
        cfg = filter_cfg or FilterConfig()
        dense_points, finfo = filter_completion_spurious(dense_points, data, cfg=cfg)
        print(
            f"[comp filter] n {finfo.get('n_in', 0)} -> {finfo.get('n_out', 0)}  "
            f"drop {finfo.get('drop_ratio', 0):.1%}"
        )

    out_abs = os.path.abspath(out_npy_path)
    out_dir = os.path.dirname(out_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(out_abs, dense_points.astype(np.float32))
    if save_unfiltered and unf is not None:
        ubase, uext = os.path.splitext(out_abs)
        raw_path = f"{ubase}_unfiltered{uext}" if uext else out_abs + "_unfiltered.npy"
        np.save(raw_path, unf.astype(np.float32))
        print(f"未过滤: {raw_path}")

    print(
        f"补全完成。输入点数: {data.shape[0]}, 输出点数: {dense_points.shape[0]}\n"
        f"已保存: {out_abs}"
    )

    if export_stages_dir:
        _export_stage_outputs(
            stage_outputs, os.path.abspath(export_stages_dir), stage_stem
        )

    if show_vis:
        pcd_in = o3d.geometry.PointCloud()
        pcd_in.points = o3d.utility.Vector3dVector(data)
        pcd_in.paint_uniform_color([1, 0, 0])
        pcd_out = o3d.geometry.PointCloud()
        pcd_out.points = o3d.utility.Vector3dVector(dense_points)
        pcd_out.paint_uniform_color([0, 1, 0])
        o3d.visualization.draw_geometries(
            [pcd_in, pcd_out.translate([1.5, 0, 0])],
            window_name="SnowflakeNet (左:输入, 右:输出)",
        )

    return dense_points
