"""
补全点云与参考点云的 FPFH 粗配准 + ICP 精配准。

典型用法：source = 网络补全后的点云，target = 同尺度的 GT / 模板点云。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d

from src.pose_estimation.fpfh import (
    global_registration_fpfh_ransac,
    numpy_to_point_cloud,
    voxel_downsample_with_fpfh,
)
from src.pose_estimation.icp import icp_refine


@dataclass
class FpfhIcpResult:
    """配准结果：最终变换 + 各阶段 Open3D 结果（含 fitness、RMSE 等）。"""

    transformation: np.ndarray  # 4x4，作用在 source：p_tgt ≈ R @ p_src + t
    ransac: o3d.pipelines.registration.RegistrationResult
    icp: o3d.pipelines.registration.RegistrationResult


def register_point_cloud_pair(
    source_points: np.ndarray,
    target_points: np.ndarray,
    voxel_size: float = 0.03,
    icp_correspondence_distance: Optional[float] = None,
    icp_mode: str = "point_to_plane",
    icp_max_iteration: int = 30,
) -> FpfhIcpResult:
    """
    对两组点云做 FPFH+RANSAC 粗配准，再 ICP 精配准。

    Args:
        source_points: 补全点云 (N, 3)
        target_points: 参考点云 (M, 3)
        voxel_size: 体素边长，需与点云尺度匹配（归一化到单位球附近时常用 0.02~0.05）
        icp_correspondence_distance: ICP 最大对应距离；默认 voxel_size * 0.4
        icp_mode: 'point_to_plane'（推荐）或 'point_to_point'
    """
    src_pcd = numpy_to_point_cloud(source_points)
    tgt_pcd = numpy_to_point_cloud(target_points)

    src_down, src_fpfh = voxel_downsample_with_fpfh(src_pcd, voxel_size)
    tgt_down, tgt_fpfh = voxel_downsample_with_fpfh(tgt_pcd, voxel_size)

    ransac = global_registration_fpfh_ransac(
        src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size
    )

    icp_dist = (
        icp_correspondence_distance
        if icp_correspondence_distance is not None
        else voxel_size * 0.4
    )
    icp_res = icp_refine(
        src_pcd,
        tgt_pcd,
        ransac.transformation,
        icp_dist,
        mode=icp_mode,
        max_iteration=icp_max_iteration,
    )

    return FpfhIcpResult(
        transformation=np.asarray(icp_res.transformation, dtype=np.float64),
        ransac=ransac,
        icp=icp_res,
    )


def transform_points(points: np.ndarray, transformation: np.ndarray) -> np.ndarray:
    """将 (N,3) 点云左乘 4x4 刚体变换。"""
    T = np.asarray(transformation, dtype=np.float64)
    pts = np.asarray(points, dtype=np.float64)
    R = T[:3, :3]
    t = T[:3, 3]
    return (pts @ R.T) + t


__all__ = [
    "FpfhIcpResult",
    "register_point_cloud_pair",
    "transform_points",
    "numpy_to_point_cloud",
    "voxel_downsample_with_fpfh",
    "global_registration_fpfh_ransac",
    "icp_refine",
]
