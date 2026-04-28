import numpy as np
import open3d as o3d

def normalize_point_cloud(pcd):
    # 1. 确保点云有法向量，如果没有则估算
    if not pcd.has_normals():
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    """居中并缩放至单位球"""
    pcd.translate(-pcd.get_center())
    points = np.asarray(pcd.points)
    normals = np.asarray(pcd.normals) # 提取法向量 [N, 3]
    
    dist = np.linalg.norm(points, axis=1)
    scale = np.max(dist) if np.max(dist) > 0 else 1.0
    points = (points / scale).astype(np.float32)
    
    return points, normals

def apply_random_hole(points, hole_radius=0.15, num_holes=2):
    """
    挖多个小洞，避免单次挖洞半径过大导致物体断裂。
    """
    corrupted_points = points.copy()
    
    for _ in range(num_holes):
        if len(corrupted_points) < 500: break # 防止挖没了
        
        idx = np.random.randint(len(corrupted_points))
        center = corrupted_points[idx]
        dists = np.linalg.norm(corrupted_points - center, axis=1)
        corrupted_points = corrupted_points[dists > hole_radius]
        
    # 补齐点数至 2048
    if len(corrupted_points) < 2048:
        fill_idx = np.random.choice(len(corrupted_points), 2048 - len(corrupted_points))
        corrupted_points = np.concatenate([corrupted_points, corrupted_points[fill_idx]], axis=0)
        
    return corrupted_points.astype(np.float32)

def apply_depth_dropout(
    points, normals, camera_pos, missing_rate=0.3, *, noise_scale: float = 0.1
):
    """
    模拟强逆光/远端信噪比降低导致的深度点云大面积丢失。
    距离越远的点，被丢弃的概率越高（概率加权随机采样，非硬截断）。

    :param points: 输入点云坐标 (N, 3)，Numpy 数组。
    :param normals: 输入点云法向量 (N, 3)，跟随对应点一起过滤。
    :param camera_pos: 相机视点位置，Numpy 数组 (如 np.array([2,2,2]))。
    :param missing_rate: 期望缺失率 (0.0~0.95)，远端点优先丢弃。
    :param noise_scale: 对归一化距离加噪的尺度；默认 0.1 与历史行为一致。HPR 后
        点云在深度上动态范围很窄，可视调试时可用 0.02~0.05 让「远端优先丢」更分明。
    :return: 缺失后的点云坐标 (M, 3), 缺失后的法向量 (M, 3)
    """
    # 限制 missing_rate 范围，杜绝空数组
    missing_rate = np.clip(missing_rate, 0.0, 0.95)

    if missing_rate <= 0.0:
        return points, normals
    num_points = len(points)
    num_keep = int(num_points * (1.0 - missing_rate))

   # 1. 计算每个点到相机的欧氏距离，归一化到 [0, 1]
    distances = np.linalg.norm(points - camera_pos, axis=1)
    distances_normalized = distances / distances.max()

    # 2. 加噪硬截断：距离加随机扰动，边缘自然模糊
    noise = np.random.normal(loc=0.0, scale=float(noise_scale), size=num_points)
    perturbed_distances = distances_normalized + noise

    # 3. 绝对精确截断：保留扰动距离最小的 num_keep 个点
    sorted_indices = np.argsort(perturbed_distances)
    keep_indices = sorted_indices[:num_keep]

    return points[keep_indices].astype(np.float32), normals[keep_indices].astype(np.float32)

def apply_specular_dropout(
    points, normals, camera_pos=np.array([2.0, 2.0, 2.0]),
    light_dir=np.array([1.0, -1.0, 0.5]),
    missing_rate=0.3,
    *,
    noise_scale: float = 0.1,
    specular_exponent: float = 1.0,
):
    """
    加噪硬截断（Noisy Hard-Cutoff）：按 (n·H)^k 排名，删去最「亮」的 missing_rate 比例点。

    :param noise_scale: 对排名标量加噪的尺度，默认 0.1 与历史一致。调试时降至 0.02~0.05
        可减轻截断界被噪声淹没的问题。
    :param specular_exponent: 对 ``clip(n·H,0,1)`` 的幂 k；k>1 时拉高高光与低光区的排序差。
        当 L 与视线 V 共线且法向 n 相当时，k≈1 会几乎无方差（需偏轴光向）。
    """
    if missing_rate <= 0.0 or len(points) == 0:
        return points, normals

    num_points = len(points)
    num_drop = int(num_points * missing_rate)
    num_keep = num_points - num_drop

    # 1. 物理向量计算 (Blinn-Phong 半角向量)
    V = camera_pos - points
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    L = np.array(light_dir, dtype=np.float32)
    L = L / np.linalg.norm(L)
    H = L + V
    H = H / np.linalg.norm(H, axis=1, keepdims=True)

    # 2. n·H，再取幂使排序有区分度（默认 k=1 与旧版对 n·H 排序等价）
    specular_intensity = np.sum(normals * H, axis=1)
    specular_intensity = np.clip(specular_intensity, 0, 1)
    k = max(float(specular_exponent), 1e-6)
    specular_rank = specular_intensity**k

    # 3. 边缘随机扰动（见 noise_scale）
    noise = np.random.normal(loc=0.0, scale=float(noise_scale), size=num_points)
    perturbed_intensity = specular_rank + noise

    # 4. 硬截断：保留排名最低(最暗)的 num_keep 个点
    sorted_indices = np.argsort(perturbed_intensity)
    keep_indices = sorted_indices[:num_keep]

    return points[keep_indices].astype(np.float32), normals[keep_indices].astype(np.float32)

def simulate_rgbd_single_view(points, camera_pos=np.array([0, 0, 2.0])):
    """
    模拟单视角 RGB-D 相机采集，剔除背面不可见的点。
    """
    # 将 numpy 数组转为 open3d 对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    # 估算法向量（HPR 需要计算视线角）
    pcd.estimate_normals()
    
    # HPR 算法参数
    radius = 10000  # 视野半径，设大一点确保包围物体
    
    # 执行隐藏点消除，返回可见点的索引
    _, pt_map = pcd.hidden_point_removal(camera_pos, radius)
    
    # 根据索引提取可见的“正面”点云 (即 2.5D 点云)
    visible_points = points[pt_map]
    visible_normals = np.asarray(pcd.normals)[pt_map]
    
    return visible_points, visible_normals
