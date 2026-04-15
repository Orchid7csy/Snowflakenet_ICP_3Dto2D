"""
同坐标系对比两个 .npy 点云（例如 processed_with_removal 的 input / gt）。
默认：无平移偏移，红=残缺/输入，浅绿=真值，便于观察真实重合度。
"""

import argparse
import numpy as np
import open3d as o3d


def compare_gt_input_abs_space(
    gt_path: str,
    input_path: str,
    point_size: float = 2.0,
):
    gt_data = np.load(gt_path, allow_pickle=True)
    input_data = np.load(input_path, allow_pickle=True)

    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(np.asarray(gt_data[:, :3], dtype=np.float64))
    pcd_gt.paint_uniform_color([0.25, 0.85, 0.35])

    pcd_input = o3d.geometry.PointCloud()
    pcd_input.points = o3d.utility.Vector3dVector(np.asarray(input_data[:, :3], dtype=np.float64))
    pcd_input.paint_uniform_color([1.0, 0.0, 0.0])

    print('同原点展示: 红=Input(残缺), 绿=GT（无 translate 分离）')

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='GT vs Input (absolute / same origin)', width=1200, height=800)
    vis.add_geometry(pcd_input)
    vis.add_geometry(pcd_gt)
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.background_color = np.array([0.05, 0.05, 0.08])
    vis.run()
    vis.destroy_window()


def main():
    root = '/home/csy/SnowflakeNet_FPFH_ICP'
    parser = argparse.ArgumentParser(description='同坐标系可视化两个 npy 点云')
    parser.add_argument('--gt', default=f'{root}/data/processed_with_removal/test/gt/airplane_0627_v1.npy')
    parser.add_argument('--input', default=f'{root}/data/processed_with_removal/test/input/airplane_0627_v1.npy')
    parser.add_argument('--point-size', type=float, default=2.0)
    args = parser.parse_args()
    compare_gt_input_abs_space(args.gt, args.input, point_size=args.point_size)


if __name__ == '__main__':
    main()
