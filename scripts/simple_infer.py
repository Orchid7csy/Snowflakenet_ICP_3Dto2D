"""
scripts/simple_infer.py

SnowflakeNet 单文件补全推理。从项目任意位置运行均可。

用法示例:
    python scripts/simple_infer.py \\
        -i data/processed/test/input/bottle_0363_v2.npy \\
        -c checkpoints/snet_finetune/ckpt-best_4090.pth

默认输出: data/completed/<输入文件名去扩展名>_completed.npy
"""

import argparse
import os
import sys
from collections import OrderedDict

import numpy as np
import open3d as o3d
import torch

# 项目根目录与 SnowflakeNet 官方代码根（与 02_train_completion.py 一致）
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..'))
_SNET_ROOT = os.path.join(_PROJECT_ROOT, 'Snet', 'SnowflakeNet-main')
_COMPLETED_DIR = os.path.join(_PROJECT_ROOT, 'data', 'completed')

for _p in (_PROJECT_ROOT, _SNET_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.model_completion import SnowflakeNet


def test_single_npy(npy_path, weight_path, out_npy_path, show_vis=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    map_loc = device if device.type == 'cuda' else 'cpu'

    model = SnowflakeNet(up_factors=[1, 4, 8])
    checkpoint = torch.load(weight_path, map_location=map_loc)
    state_dict = checkpoint['model']

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()

    data = np.load(npy_path).astype(np.float32)

    if data.shape[0] < 2048:
        idx = np.random.choice(data.shape[0], 2048, replace=True)
        data = data[idx]
    elif data.shape[0] > 2048:
        idx = np.random.choice(data.shape[0], 2048, replace=False)
        data = data[idx]

    input_tensor = torch.from_numpy(data).unsqueeze(0).to(device)

    with torch.no_grad():
        ret = model(input_tensor)
        dense_points = ret[-1].squeeze().cpu().numpy()

    out_abs = os.path.abspath(out_npy_path)
    out_dir = os.path.dirname(out_abs)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(out_abs, dense_points.astype(np.float32))

    print(
        f"补全完成。输入点数: {data.shape[0]}, 输出点数: {dense_points.shape[0]}\n"
        f"已保存: {out_abs}"
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


def _default_out_path(input_npy: str) -> str:
    stem, _ = os.path.splitext(os.path.basename(input_npy))
    return os.path.join(_COMPLETED_DIR, f'{stem}_completed.npy')


def main():
    parser = argparse.ArgumentParser(description='SnowflakeNet 单样本 npy 补全')
    parser.add_argument(
        '-i', '--input',
        default=os.path.join(
            _PROJECT_ROOT,
            'data/processed_with_removal/test/input/airplane_0627_v1.npy',
        ),
        help='输入残缺点云 .npy (N,3)',
    )
    parser.add_argument(
        '-c', '--ckpt',
        default=os.path.join(
            _PROJECT_ROOT,
            'checkpoints/snet_finetune/ckpt-airplane-best.pth',
        ),
        help='模型权重路径',
    )
    parser.add_argument(
        '-o', '--out',
        default=None,
        help='补全结果保存路径 (.npy)；默认 data/completed/<输入主文件名>_completed.npy',
    )
    parser.add_argument(
        '--no-vis',
        action='store_true',
        help='不弹出 Open3D 可视化窗口',
    )
    args = parser.parse_args()

    out_path = args.out if args.out else _default_out_path(args.input)
    test_single_npy(args.input, args.ckpt, out_path, show_vis=not args.no_vis)


if __name__ == '__main__':
    main()
