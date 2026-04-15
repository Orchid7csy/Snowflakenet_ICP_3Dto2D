"""
scripts/04_check_alignment.py

用法：
    python scripts/04_check_alignment.py --data_root data/processed --num_samples 6

检查内容：
    1. input / gt 点数统计
    2. 两者几何中心偏移量（绝对对齐检查）
    3. 两者包围盒尺度比（归一化一致性检查）
    4. Open3D 可视化：红=input，绿=gt，叠加显示
"""

import argparse
import os
import random
import numpy as np
import open3d as o3d

# ── 参数 ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--data_root", type=str, default="/home/csy/SnowflakeNet_FPFH_ICP/data/processed")
parser.add_argument("--num_samples", type=int, default=6)
parser.add_argument("--no_vis", action="store_true", help="只打印统计，不弹窗")
args = parser.parse_args()

# ── 加载函数（按你的实际存储格式二选一）────────────────────────────────────────
def load_pcd(path: str) -> np.ndarray:
    """返回 (N, 3) float32。支持 .npy / .npz / .txt"""
    ext = os.path.splitext(path)[-1]
    if ext == ".npy":
        return np.load(path).astype(np.float32)
    elif ext == ".npz":
        d = np.load(path)
        key = list(d.keys())[0]          # 取第一个数组
        return d[key].astype(np.float32)
    elif ext == ".txt":
        return np.loadtxt(path, dtype=np.float32)
    else:
        raise ValueError(f"不支持的格式: {ext}")

def find_pairs(data_root: str):
    """
    实际结构：
        data_root/
            train/
                input/  <- 所有 input 文件
                gt/     <- 所有 gt 文件（文件名与 input 一一对应）
            test/
                input/
                gt/
    """
    pairs = []
    for split in ["train", "test"]:
        inp_dir = os.path.join(data_root, split, "input")
        gt_dir  = os.path.join(data_root, split, "gt")

        if not os.path.isdir(inp_dir) or not os.path.isdir(gt_dir):
            continue

        for fname in sorted(os.listdir(inp_dir)):
            inp_path = os.path.join(inp_dir, fname)
            gt_path  = os.path.join(gt_dir, fname)   # 假设文件名完全一致

            if not os.path.isfile(gt_path):
                print(f"[警告] 找不到对应 gt：{gt_path}，跳过")
                continue

            label = f"{split}/{fname}"
            pairs.append((inp_path, gt_path, label))

    return pairs

# ── 统计函数 ──────────────────────────────────────────────────────────────────
def bbox_diag(pts: np.ndarray) -> float:
    return float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))

def analyze(inp: np.ndarray, gt: np.ndarray, label: str):
    center_inp = inp.mean(axis=0)
    center_gt  = gt.mean(axis=0)
    offset     = np.linalg.norm(center_inp - center_gt)
    diag_inp   = bbox_diag(inp)
    diag_gt    = bbox_diag(gt)
    scale_ratio = diag_inp / (diag_gt + 1e-8)

    # 判断是否对齐（经验阈值）
    offset_ok = offset < 0.05        # 中心偏移 < 包围盒的 5%
    scale_ok  = 0.8 < scale_ratio < 1.2

    print(f"\n{'='*60}")
    print(f"样本: {label}")
    print(f"  input 点数       : {len(inp)}")
    print(f"  gt    点数       : {len(gt)}")
    print(f"  中心偏移 (L2)    : {offset:.4f}  {'✓' if offset_ok else '✗ 偏移过大！'}")
    print(f"  input 包围盒对角 : {diag_inp:.4f}")
    print(f"  gt    包围盒对角 : {diag_gt:.4f}")
    print(f"  尺度比 inp/gt    : {scale_ratio:.4f}  {'✓' if scale_ok else '✗ 尺度不一致！'}")
    print(f"  input 中心       : [{center_inp[0]:.3f}, {center_inp[1]:.3f}, {center_inp[2]:.3f}]")
    print(f"  gt    中心       : [{center_gt[0]:.3f}, {center_gt[1]:.3f}, {center_gt[2]:.3f}]")

    return offset_ok and scale_ok

def make_o3d_pcd(pts: np.ndarray, color) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.paint_uniform_color(color)
    return pcd

# ── 主流程 ────────────────────────────────────────────────────────────────────
pairs = find_pairs(args.data_root)
if not pairs:
    print(f"[错误] 在 {args.data_root} 下未找到任何 input/gt 配对，请检查路径和目录结构。")
    exit(1)

print(f"共找到 {len(pairs)} 个样本，随机抽取 {args.num_samples} 个。")
sampled = random.sample(pairs, min(args.num_samples, len(pairs)))

all_ok = True
for inp_path, gt_path, label in sampled:
    inp = load_pcd(inp_path)
    gt  = load_pcd(gt_path)
    ok  = analyze(inp, gt, label)
    all_ok = all_ok and ok

    if not args.no_vis:
        pcd_inp = make_o3d_pcd(inp, [0.9, 0.2, 0.2])  # 红：input
        pcd_gt  = make_o3d_pcd(gt,  [0.2, 0.8, 0.2])  # 绿：gt

        # 坐标轴：X=红 Y=绿 Z=蓝，size 按 gt 包围盒自适应
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=bbox_diag(gt) * 0.3, origin=[0, 0, 0]
        )

        print("\n  [可视化] 红=input  绿=gt  关闭窗口后继续下一个样本")
        o3d.visualization.draw_geometries(
            [pcd_inp, pcd_gt, axis],
            window_name=os.path.basename(label),
            width=1000, height=700
        )

print(f"\n{'='*60}")
if all_ok:
    print("总结：所有抽查样本对齐正常，空间锚定不是主要问题。")
else:
    print("总结：存在对齐异常样本，建议检查 01_preprocessing.py 中")
    print("      normalize_point_cloud 与光照截断的执行顺序。")