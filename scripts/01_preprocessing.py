import os
os.environ["OPEN3D_HEADLESS"] = "1"  # 禁用 GUI 初始化，放在所有 import 之前
import multiprocessing
import open3d as o3d
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import tempfile

# 确保能导入 src 目录下的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.transforms import normalize_point_cloud, apply_random_hole, apply_specular_dropout, apply_depth_dropout

def process_single_file(args):
    import open3d as o3d
    import os
    import numpy as np
    import tempfile
    
    file_path, cat, split, output_path, num_points = args
    tmp_path = None
    
    try:
        is_repaired = False
        # 1. 读取并修复内容
        with open(file_path, 'rb') as f:
            raw_content = f.read().decode('utf-8', errors='ignore')
        
        first_line = raw_content.split('\n', 1)[0].strip()
        final_content = raw_content
        if first_line.startswith('OFF') and len(first_line) > 3 and first_line[3].isdigit():
            final_content = "OFF\n" + raw_content[3:].lstrip()
            is_repaired = True

        # 2. 稳健的临时文件创建
        _, tmp_path = tempfile.mkstemp(suffix='.off')
        
        # 3. 写入修复后的内容
        with open(tmp_path, 'w', encoding='utf-8') as f_tmp:
            f_tmp.write(final_content)
        
        # 4. Open3D 读取
        mesh = o3d.io.read_triangle_mesh(tmp_path)
        
        # 5. 读完立刻删除临时文件
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            tmp_path = None
        
        if not mesh.has_triangles(): 
            return False, file_path, False

        # --- 核心处理流水线 ---
        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        gt_points, gt_normals = normalize_point_cloud(pcd)

        base_name = os.path.basename(file_path).replace('.off', '')
        num_views = 8

        # GT：每份视角对应独立的随机 2048 子集
        idx_gt = np.random.choice(len(gt_points), 2048, replace=False)
        gt_save = gt_points[idx_gt]

        for view_idx in range(num_views):
            # 1. 随机相机视角
            theta = np.random.uniform(0, 2 * np.pi)
            phi   = np.random.uniform(0, np.pi)
            r = 3.0
            camera_pos = np.array([
                r * np.sin(phi) * np.cos(theta),
                r * np.sin(phi) * np.sin(theta),
                r * np.cos(phi)
            ])

            # 2. HPR：提取可见点
            pcd_normalized = o3d.geometry.PointCloud()
            pcd_normalized.points = o3d.utility.Vector3dVector(gt_points)
            _, pt_map = pcd_normalized.hidden_point_removal(camera_pos, 100)

            visible_points  = gt_points[pt_map]
            visible_normals = gt_normals[pt_map]

            # 3. 光照退化（随机选 A / B 两种场景）
            expected_rate = 0.2

            if np.random.rand() > 0.5:
                # 场景 A：强逆光 / 宽动态范围失效（距离概率丢失）
                corrupted, _ = apply_depth_dropout(
                    visible_points,
                    visible_normals,
                    camera_pos=camera_pos,
                    missing_rate=expected_rate
                )
            else:
                # 场景 B：定向强光 / 高光致盲（法向量半角概率丢失）
                theta_l = np.random.uniform(0, 2 * np.pi)
                phi_l   = np.random.uniform(0, np.pi)
                light_dir = np.array([
                    np.sin(phi_l) * np.cos(theta_l),
                    np.sin(phi_l) * np.sin(theta_l),
                    np.cos(phi_l)
                ])
                corrupted, _ = apply_specular_dropout(
                    visible_points,
                    visible_normals,
                    camera_pos=camera_pos,
                    light_dir=light_dir,
                    missing_rate=expected_rate
                )

            # 4. 高斯噪声
            noise = np.random.normal(0, 0.002, corrupted.shape).astype(np.float32)
            corrupted = corrupted + noise

            # 5. 采样
            file_name = f"{base_name}_v{view_idx}.npy"

            # Input：从退化后剩余点抽 2048
            n_corrupted = corrupted.shape[0]
            if n_corrupted >= 2048:
                idx_in = np.random.choice(n_corrupted, 2048, replace=False)
            else:
                idx_in = np.random.choice(n_corrupted, 2048, replace=True)
            corrupted_save = corrupted[idx_in]

            # 6. 保存
            np.save(os.path.join(output_path, split, 'gt',    file_name), gt_save)
            np.save(os.path.join(output_path, split, 'input', file_name), corrupted_save)

        return True, file_path, is_repaired

    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            try: os.remove(tmp_path)
            except: pass
        print(f"处理 {file_path} 出错: {e}")
        return False, file_path, False

def main():
    # 1. 路径配置
    RAW_DIR = "/home/csy/SnowflakeNet_FPFH_ICP/data/raw/ModelNet40"
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.abspath(os.path.join(current_script_path, "../data/processed"))

    if not os.path.exists(RAW_DIR):
        print(f"❌ 原始数据集路径不存在: {RAW_DIR}")
        return

    # 2. 扫描所有 .off 文件
    tasks = []
    print(f"🔍 正在扫描数据集: {RAW_DIR} ...")
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if f.endswith('.off'):
                file_path = os.path.join(root, f)
                
                # 从路径解析 category (cat) 和 split (train/test)
                # 预期的路径结构: .../ModelNet40/category/split/file.off
                parts = file_path.replace('\\', '/').split('/')
                try:
                    idx = parts.index('ModelNet40')
                    cat = parts[idx + 1]
                    split = parts[idx + 2]
                    
                    # 创建对应的输出目录
                    os.makedirs(os.path.join(OUTPUT_DIR, split, 'gt'), exist_ok=True)
                    os.makedirs(os.path.join(OUTPUT_DIR, split, 'input'), exist_ok=True)
                    
                    tasks.append((file_path, cat, split, OUTPUT_DIR, 8192))
                except (ValueError, IndexError):
                    continue

    print(f"共发现 {len(tasks)} 个待处理模型。")

    # 3. 多进程并行处理 (利用 ThreadPoolExecutor 或 ProcessPoolExecutor)
    # 注意：如果 process_single_file 内部涉及大量 CPU 运算，推荐用 ProcessPoolExecutor
    print(f"🚀 开始批处理预处理 (使用 8 个线程)...")
    results = []
    with ProcessPoolExecutor(max_workers=4) as executor:
        results = list(tqdm(executor.map(process_single_file, tasks), total=len(tasks)))

    # 4. 统计结果
    success_count = sum(1 for r in results if r[0])
    repaired_count = sum(1 for r in results if r[2])
    print("\n" + "=" * 40)
    print(f"✅ 处理完成！")
    print(f"📊 成功: {success_count} / {len(tasks)}")
    print(f"🔧 修复 Header: {repaired_count}")
    print(f"📂 输出目录: {OUTPUT_DIR}")
    print("=" * 40)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    main()