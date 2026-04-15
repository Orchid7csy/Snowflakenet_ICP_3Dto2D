import open3d as o3d
import numpy as np
import os
import sys
import tempfile

# 确保能导入 src 目录下的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.transforms import normalize_point_cloud, apply_specular_dropout, apply_depth_dropout

def show_pcd(points, window_name, color):
    """可视化辅助函数，打印当前点数并弹窗展示"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.paint_uniform_color(color)
    print(f"[{window_name}] 当前点云数量: {len(points)}")
    o3d.visualization.draw_geometries([pcd], window_name=window_name)

def run_ablation(file_path):
    print(f"🚀 开始消融实验: {os.path.basename(file_path)}")
    
    # 0. 修复 Header 并读取
    with open(file_path, 'rb') as f:
        raw_content = f.read().decode('utf-8', errors='ignore')
    first_line = raw_content.split('\n', 1)[0].strip()
    if first_line.startswith('OFF') and len(first_line) > 3 and first_line[3].isdigit():
        raw_content = "OFF\n" + raw_content[3:].lstrip()
    
    _, tmp_path = tempfile.mkstemp(suffix='.off')
    with open(tmp_path, 'w', encoding='utf-8') as f_tmp:
        f_tmp.write(raw_content)
    
    mesh = o3d.io.read_triangle_mesh(tmp_path)
    os.remove(tmp_path)
    
    if not mesh.has_triangles():
        print("❌ 模型无面片，无法采样！")
        return

    # ==========================================
    # 步骤 1: 基础采样与归一化 (GT)
    # ==========================================
    pcd = mesh.sample_points_poisson_disk(2048)
    gt_points, gt_normals = normalize_point_cloud(pcd)
    show_pcd(gt_points, "Step 1: GT 原始完整点云", [0, 1, 0]) # 绿色

    # ==========================================
    # 步骤 2: HPR 单视角可见性
    # ==========================================
    camera_pos = np.array([2.0, 2.0, 2.0])
    pcd_gt = o3d.geometry.PointCloud()
    pcd_gt.points = o3d.utility.Vector3dVector(gt_points)
    _, pt_map = pcd_gt.hidden_point_removal(camera_pos, 100)
    
    hpr_points = gt_points[pt_map]
    hpr_normals = gt_normals[pt_map]
    show_pcd(hpr_points, "Step 2: 经过 HPR 单视角遮挡", [1, 0.5, 0]) # 橙色

    # ==========================================
    # 步骤 3: 物理退化 (二选一，你可以注释掉其中一个观察)
    # ==========================================
    expected_rate = 0.3 # 设定 30% 的期望缺失率
    
    # --- 分支 A: 强逆光深度丢失 ---
    degraded_points, degraded_normals = apply_depth_dropout(
        hpr_points, hpr_normals, camera_pos, missing_rate=expected_rate)
    show_pcd(degraded_points, f"Step 3A: 逆光深度丢失 (设定率 {expected_rate})", [0, 0, 1]) # 蓝色

    # --- 分支 B: 高反光定向丢失 ---
    # light_dir = np.array([1.0, -1.0, 0.5])
    # degraded_points, degraded_normals = apply_specular_dropout(
    #     hpr_points, hpr_normals, camera_pos, light_dir, missing_rate=expected_rate)
    # show_pcd(degraded_points, f"Step 3B: 定向高光剔除 (设定率 {expected_rate})", [1, 0, 0]) # 红色

    # ==========================================
    # 步骤 4: 传感器噪声
    # ==========================================
    noise = np.random.normal(0, 0.01, degraded_points.shape).astype(np.float32)
    final_points = degraded_points + noise
    show_pcd(final_points, "Step 4: 叠加高斯噪声 (最终 Input)", [0.5, 0.5, 0.5]) # 灰色

if __name__ == "__main__":
    # 找一个你刚才觉得“碎成了渣”的飞机 off 文件路径填进这里
    test_file = "/home/csy/SnowflakeNet_FPFH_ICP/data/raw/ModelNet40/airplane/test/airplane_0641.off"
    run_ablation(test_file)