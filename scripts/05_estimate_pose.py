"""
基于补全点云与参考点云（如 GT）的 FPFH 粗配准 + ICP 精配准。

用法（在项目根目录）:
    python scripts/05_estimate_pose.py \\
        --source path/to/completed.npy \\
        --target path/to/gt.npy \\
        --voxel_size 0.03

可选：--save_T pose.npy 保存 4x4 变换；--vis 可视化对齐结果。
"""

import argparse
import os
import sys

import numpy as np
import open3d as o3d

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.pose_estimation import register_point_cloud_pair, transform_points


def load_points(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[-1].lower()
    if ext == ".npy":
        return np.load(path).astype(np.float64)
    if ext == ".npz":
        d = np.load(path)
        key = list(d.keys())[0]
        return d[key].astype(np.float64)
    if ext == ".txt":
        return np.loadtxt(path, dtype=np.float64)
    raise ValueError(f"不支持的格式: {ext}")


def main():
    parser = argparse.ArgumentParser(description="FPFH + ICP 点云配准")
    parser.add_argument("--source", type=str, required=True, help="补全/待对齐点云 (.npy)")
    parser.add_argument("--target", type=str, required=True, help="参考点云 (.npy)，如 GT")
    parser.add_argument("--voxel_size", type=float, default=0.03, help="FPFH 下采样体素，与点云尺度匹配")
    parser.add_argument(
        "--icp_dist",
        type=float,
        default=None,
        help="ICP 最大对应距离，默认 voxel_size * 0.4",
    )
    parser.add_argument(
        "--icp_mode",
        type=str,
        default="point_to_plane",
        choices=["point_to_plane", "point_to_point"],
    )
    parser.add_argument("--save_T", type=str, default=None, help="将 4x4 变换保存为 .npy")
    parser.add_argument("--vis", action="store_true", help="可视化（需图形环境）")
    args = parser.parse_args()

    src = load_points(args.source)
    tgt = load_points(args.target)

    out = register_point_cloud_pair(
        src,
        tgt,
        voxel_size=args.voxel_size,
        icp_correspondence_distance=args.icp_dist,
        icp_mode=args.icp_mode,
    )

    print("=== RANSAC (FPFH) ===")
    print(f"  fitness: {out.ransac.fitness:.6f}")
    print(f"  inlier_rmse: {out.ransac.inlier_rmse:.6f}")
    print("=== ICP ===")
    print(f"  fitness: {out.icp.fitness:.6f}")
    print(f"  inlier_rmse: {out.icp.inlier_rmse:.6f}")
    print("=== T (source -> target) ===")
    print(out.transformation)

    if args.save_T:
        np.save(args.save_T, out.transformation)
        print(f"已保存: {args.save_T}")

    if args.vis:
        pcd_s = o3d.geometry.PointCloud()
        pcd_s.points = o3d.utility.Vector3dVector(transform_points(src, out.transformation))
        pcd_s.paint_uniform_color([1.0, 0.7, 0.0])
        pcd_t = o3d.geometry.PointCloud()
        pcd_t.points = o3d.utility.Vector3dVector(tgt)
        pcd_t.paint_uniform_color([0.0, 0.65, 0.93])
        o3d.visualization.draw_geometries(
            [pcd_s, pcd_t],
            window_name="对齐后: 黄=变换后 source, 青=target",
            width=1000,
            height=700,
        )


if __name__ == "__main__":
    main()
