import open3d as o3d
import numpy as np
import os
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def process_single_file(args):
    """单个文件处理函数，供进程池调用"""
    file_path, cat, split, output_path, num_points = args
    
    try:
        # 1. 读取 Mesh
        mesh = o3d.io.read_triangle_mesh(file_path)
        if not mesh.has_triangles():
            return False

        # 2. 泊松盘采样 (高质量但耗时)
        pcd = mesh.sample_points_poisson_disk(number_of_points=num_points)
        
        # 3. 归一化
        pcd.translate(-pcd.get_center())
        points = np.asarray(pcd.points)
        dist = np.linalg.norm(points, axis=1)
        scale = np.max(dist) if np.max(dist) > 0 else 1.0
        points = (points / scale).astype(np.float32)
        
        # 4. 生成模拟极端条件的噪声版本
        noise = np.random.normal(0, 0.02, points.shape).astype(np.float32)
        corrupted = points + noise
        
        # 5. 保存
        file_name = os.path.basename(file_path).replace('.off', '.npy')
        save_name = f"{cat}_{file_name}"
        
        np.save(os.path.join(output_path, split, 'gt', save_name), points)
        np.save(os.path.join(output_path, split, 'input', save_name), corrupted)
        return True
    except Exception as e:
        print(f"处理 {file_path} 出错: {e}")
        return False

def batch_preprocess_parallel(base_path, output_path, categories, num_points=2048):
    # 准备任务列表
    tasks = []
    for cat in categories:
        for split in ['train', 'test']:
            # 创建目录
            os.makedirs(os.path.join(output_path, split, 'gt'), exist_ok=True)
            os.makedirs(os.path.join(output_path, split, 'input'), exist_ok=True)
            
            src_dir = os.path.join(base_path, cat, split)
            if not os.path.exists(src_dir): continue
            
            for f in os.listdir(src_dir):
                if f.endswith('.off'):
                    tasks.append((os.path.join(src_dir, f), cat, split, output_path, num_points))

    print(f"🚀 准备并行处理 {len(tasks)} 个文件...")
    
    # 使用进程池执行
    # max_workers 默认使用 CPU 核心数
    with ProcessPoolExecutor() as executor:
        # 使用 list() 强制迭代完成，结合 tqdm 显示总进度
        results = list(tqdm(executor.map(process_single_file, tasks), total=len(tasks)))

    success_count = sum(results)
    print(f"\n✅ 处理完成！成功: {success_count}, 失败: {len(tasks) - success_count}")

if __name__ == "__main__":
    BASE_DIR = "/home/csy/graduation_project/assets/ModelNet40"
    OUTPUT_DIR = "/home/csy/graduation_project/assets/processed_data"
    TARGET_CATS = ['airplane', 'car', 'bottle'] # 你可以继续增加类别
    
    batch_preprocess_parallel(BASE_DIR, OUTPUT_DIR, TARGET_CATS)