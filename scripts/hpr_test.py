import open3d as o3d
import numpy as np
import os

def check_hpr_effect():
    # 获取当前脚本所在的绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 动态拼接，这样无论你在哪运行，它都能准确找到上一层的 data
    base_dir = os.path.abspath(os.path.join(current_dir, "../data/raw/ModelNet40"))
    test_samples = [
        ('airplane', 'airplane/test/airplane_0627.off'),
        ('chair', 'chair/test/chair_0891.off'),
        ('monitor', 'monitor/test/monitor_0565.off')
    ]

    for cat, rel_path in test_samples:
        full_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(full_path):
            print(f"找不到文件: {full_path}，请确认路径")
            continue

        # 2. 读取并采样
        mesh = o3d.io.read_triangle_mesh(full_path)
        # 检查是否有三角形面片
        if len(mesh.triangles) == 0:
            print(f"⚠️ {cat} 模型没有面片，改用直接点云转换...")
            # 如果没面，直接把 mesh 里的顶点当成点云
            pcd = o3d.geometry.PointCloud()
            pcd.points = mesh.vertices
        else:
            # 如果有面，执行高质量 Poisson 采样
            pcd = mesh.sample_points_poisson_disk(2048)
        
        # 归一化到单位球 (这一步很重要，决定了相机距离)
        pcd.translate(-pcd.get_center())
        pcd.scale(1 / np.max(np.abs(pcd.get_max_bound())), center=(0, 0, 0))

        # 3. HPR 参数测试 (相机设在斜上方 [2, 2, 2])
        camera_pos = np.array([2.0, 2.0, 2.0])
        radius = 100 # 足够大的半径确保采样稳定
        _, pt_map = pcd.hidden_point_removal(camera_pos, radius)
        
        # 4. 区分颜色：原始点云绿色，采样出的 2.5D 红色
        pcd_25d = pcd.select_by_index(pt_map)
        pcd.paint_uniform_color([0, 1, 0])      # 绿色 (GT)
        pcd_25d.paint_uniform_color([1, 0, 0])  # 红色 (2.5D Input)

        print(f"正在显示类别: {cat} | 红色为相机可视面")
        # 弹窗显示，你可以用鼠标旋转看效果
        o3d.visualization.draw_geometries([pcd_25d], window_name=f"HPR Effect: {cat}")

if __name__ == "__main__":
    check_hpr_effect()