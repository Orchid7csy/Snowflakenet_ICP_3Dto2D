"""
FPFH 特征与基于 RANSAC 的全局配准（粗对齐）。

适用于补全点云与参考点云（如 GT 或模板）在同一度量尺度下、存在较大初始位姿差的情况。
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import open3d as o3d


def _registration_api():
    """
    Open3D API 兼容层：
    - 新版 (>=0.10): ``o3d.pipelines.registration``
    - 旧版 (0.9):    ``o3d.registration``
    """
    if hasattr(o3d, "pipelines") and hasattr(o3d.pipelines, "registration"):
        return o3d.pipelines.registration
    if hasattr(o3d, "registration"):
        return o3d.registration
    raise AttributeError(
        "Open3D 中未找到 registration API（既无 pipelines.registration 也无 registration）"
    )


def numpy_to_point_cloud(points: np.ndarray) -> o3d.geometry.PointCloud:
    """(N, 3) float -> Open3D PointCloud。"""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points 形状应为 (N, 3)，当前 {pts.shape}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def voxel_downsample_with_fpfh(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
    normal_radius: Optional[float] = None,
    fpfh_radius: Optional[float] = None,
    max_nn_normal: int = 30,
    max_nn_fpfh: int = 100,
) -> Tuple[o3d.geometry.PointCloud, Any]:
    """
    体素下采样，估计法向，计算 FPFH。

    normal_radius / fpfh_radius 默认按 voxel_size 的倍数设定，与 Open3D 教程一致。
    """
    if voxel_size <= 0:
        raise ValueError("voxel_size 必须为正数")

    reg = _registration_api()
    pcd_down = pcd.voxel_down_sample(voxel_size)
    if len(pcd_down.points) < 10:
        raise ValueError(f"下采样后点数过少 ({len(pcd_down.points)})，请减小 voxel_size")

    nr = normal_radius if normal_radius is not None else voxel_size * 2.0
    fr = fpfh_radius if fpfh_radius is not None else voxel_size * 5.0

    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=nr, max_nn=max_nn_normal)
    )
    fpfh = reg.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=fr, max_nn=max_nn_fpfh),
    )
    return pcd_down, fpfh


def global_registration_fpfh_ransac(
    source_down: o3d.geometry.PointCloud,
    target_down: o3d.geometry.PointCloud,
    source_fpfh: Any,
    target_fpfh: Any,
    voxel_size: float,
    mutual_filter: bool = True,
    ransac_n: int = 4,
    max_iterations: int = 4_000_000,
    confidence: int = 500,
    edge_length_ratio: float = 0.9,
) -> Any:
    """
    基于 FPFH 描述子匹配的 RANSAC，得到 source -> target 的刚体初值。

    返回的 transformation 作用在 source 上：p_target ≈ R @ p_source + t。
    """
    reg = _registration_api()
    distance_threshold = voxel_size * 1.5

    checkers = [
        reg.CorrespondenceCheckerBasedOnEdgeLength(edge_length_ratio),
        reg.CorrespondenceCheckerBasedOnDistance(distance_threshold),
    ]
    # 旧版 (o3d 0.9) RANSACConvergenceCriteria 只收 (max_iteration, max_validation)，
    # 且 registration_ransac_based_on_feature_matching 签名不含 ``mutual_filter``。
    is_new_api = hasattr(o3d, "pipelines") and hasattr(o3d.pipelines, "registration")
    estimation = reg.TransformationEstimationPointToPoint(False)

    if is_new_api:
        criteria = reg.RANSACConvergenceCriteria(max_iterations, confidence)
        result = reg.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            mutual_filter,
            distance_threshold,
            estimation,
            ransac_n,
            checkers,
            criteria,
        )
    else:
        criteria = reg.RANSACConvergenceCriteria(max_iterations, confidence)
        result = reg.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            distance_threshold,
            estimation,
            ransac_n,
            checkers,
            criteria,
        )
    return result
