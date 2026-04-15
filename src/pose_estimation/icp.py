"""
ICP 精对齐：在粗初值（如 FPFH+RANSAC）基础上细化刚体变换。
"""

from __future__ import annotations

import numpy as np
import open3d as o3d


def _registration_api():
    """
    Open3D API 兼容层：
    - 新版：o3d.pipelines.registration
    - 旧版：o3d.registration
    """
    if hasattr(o3d, "pipelines") and hasattr(o3d.pipelines, "registration"):
        return o3d.pipelines.registration
    if hasattr(o3d, "registration"):
        return o3d.registration
    raise AttributeError("Open3D 中未找到 registration API（既无 pipelines.registration 也无 registration）")


def estimate_normals_hybrid(
    pcd: o3d.geometry.PointCloud,
    radius: float,
    max_nn: int = 30,
) -> None:
    """为点云估计法向（原地修改）。"""
    pcd.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )


def icp_refine(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    init_transform: np.ndarray,
    max_correspondence_distance: float,
    mode: str = "point_to_plane",
    max_iteration: int = 30,
) -> o3d.pipelines.registration.RegistrationResult:
    """
    ICP 细化。mode: 'point_to_point' | 'point_to_plane'。

    point_to_plane 需要双方已估计法向；本函数会在缺失时按半径自动估计。
    """
    if init_transform.shape != (4, 4):
        raise ValueError("init_transform 须为 4x4 齐次矩阵")

    reg = _registration_api()
    src = source
    tgt = target

    if mode == "point_to_plane":
        if not src.has_normals():
            estimate_normals_hybrid(src, max_correspondence_distance * 2.0)
        if not tgt.has_normals():
            estimate_normals_hybrid(tgt, max_correspondence_distance * 2.0)
        estimation = reg.TransformationEstimationPointToPlane()
    elif mode == "point_to_point":
        estimation = reg.TransformationEstimationPointToPoint()
    else:
        raise ValueError("mode 应为 'point_to_point' 或 'point_to_plane'")

    criteria = reg.ICPConvergenceCriteria(
        relative_fitness=1e-6,
        relative_rmse=1e-6,
        max_iteration=max_iteration,
    )

    return reg.registration_icp(
        src,
        tgt,
        max_correspondence_distance,
        init_transform,
        estimation,
        criteria,
    )
