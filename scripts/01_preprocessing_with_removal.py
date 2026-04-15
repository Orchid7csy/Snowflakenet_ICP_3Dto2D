"""
01_preprocessing_with_removal.py

在 01_preprocessing 流水线基础上：小角度随机旋转 + 平移、残缺包围盒中心化、PCA 主轴对齐。

每个 split 下输出四个目录（同名 stem）：
  obs/   世界系残缺观测 p_obs_w（ICP / 位姿 Target）
  input/ 归一化补全网络输入（−C_bbox 与 R_pca 后）
  gt/    与 input 同系的监督真值（T_rand 后 −C_bbox 与 R_pca）
  meta/  np.savez：R_rand, t_rand, C_bbox, R_pca, mu_pca，便于逆变换回物理系评估 6DoF

可视化: python3 scripts/01_preprocessing_with_removal.py --vis

行向量约定下与 meta 一致的逆变换（由网络/ICP 坐标 p_input 回到 p_obs_w）:
  p_norm = (p_input - mu_pca) @ R_pca + mu_pca
  p_obs_w = p_norm + C_bbox
再由 p_obs_w 与 R_rand、t_rand 可回到施加 T_rand 前的规范系点云。
"""

import os
import sys
import multiprocessing

# 必须在 import open3d 之前；子进程不得因继承 argv 而关闭 headless
if (
    '--vis' in sys.argv
    and multiprocessing.current_process().name == 'MainProcess'
):
    os.environ['PREPROCESS_WITH_REMOVAL_VIS'] = '1'
if os.environ.get('PREPROCESS_WITH_REMOVAL_VIS') != '1':
    os.environ['OPEN3D_HEADLESS'] = '1'

import argparse
import open3d as o3d
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import tempfile

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data.transforms import normalize_point_cloud, apply_specular_dropout, apply_depth_dropout


def random_rotation_matrix(max_deg: float = 15.0) -> np.ndarray:
    """
    小角度随机旋转：绕 X/Y/Z 各独立均匀采样于 [-max_deg, +max_deg]，
    合成 R = Rz @ Ry @ Rx（模拟粗对齐后的残余姿态误差）。
    """
    rx = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    ry = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    rz = np.deg2rad(np.random.uniform(-max_deg, max_deg))
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    R = (Rz @ Ry @ Rx).astype(np.float32)
    return R


def random_translation_vector(scale_min: float, scale_max: float, extent: float) -> np.ndarray:
    """
    随机平移方向，模长在 [scale_min * extent, scale_max * extent]。
    extent 取场景尺度（如 GT 点云 AABB 最大边长），使位移与归一化物体可比拟。
    """
    mag = np.random.uniform(scale_min, scale_max) * max(extent, 1e-6)
    d = np.random.randn(3).astype(np.float32)
    d /= np.linalg.norm(d) + 1e-8
    return (mag * d).astype(np.float32)


def apply_random_se3(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """P' = P @ R.T + t，points (N,3)。"""
    return (points.astype(np.float32) @ R.T + t).astype(np.float32)


def normalize_by_bbox(points: np.ndarray, use_oriented: bool = False):
    """
    仅使用包围盒几何中心 C_bbox（Open3D），严禁点云质心。
    返回 P_norm = P - C_bbox，以及 C_bbox (3,)。
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if use_oriented:
        bbox = pcd.get_oriented_bounding_box()
    else:
        bbox = pcd.get_axis_aligned_bounding_box()
    c_bbox = np.asarray(bbox.get_center(), dtype=np.float32)
    p_norm = (points.astype(np.float32) - c_bbox).astype(np.float32)
    return p_norm, c_bbox


def aabb_center(points: np.ndarray) -> np.ndarray:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    return np.asarray(pcd.get_axis_aligned_bounding_box().get_center(), dtype=np.float32)


def _orthonormal_frame_from_axis(v: np.ndarray) -> np.ndarray:
    """列向量为正交基，第一列为 v（单位化）。"""
    v = np.asarray(v, dtype=np.float64).reshape(3)
    n = np.linalg.norm(v)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    v = v / n
    tmp = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(v, tmp)) > 0.9:
        tmp = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    e2 = np.cross(v, tmp)
    e2 /= np.linalg.norm(e2) + 1e-12
    e3 = np.cross(v, e2)
    return np.stack([v, e2, e3], axis=1)


def pca_align_principal_to_axis(
    points: np.ndarray,
    target_axis: str = 'z',
    min_eigval_ratio: float = 1e-4,
):
    """
    在 normalize_by_bbox 之后：用 Open3D compute_mean_and_covariance 得协方差，
    特征分解后取最大特征值对应主轴，旋转至与 +X/+Y/+Z 对齐（右手系）。

    对点应用 p' = (p - mu) @ R.T + mu（mu 为 PCA 所用均值；与 bbox 中心不同属正常）。

    返回 (p_aligned, R, mu)；若协方差退化则 R=I。
    """
    pcd = o3d.geometry.PointCloud()
    pts = np.asarray(points, dtype=np.float64)
    pcd.points = o3d.utility.Vector3dVector(pts)
    mu, cov = pcd.compute_mean_and_covariance()
    mu = np.asarray(mu, dtype=np.float64).reshape(3)
    cov = np.asarray(cov, dtype=np.float64).reshape(3, 3)

    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    lam1 = float(evals[order[0]])
    lam2 = float(evals[order[1]]) if len(order) > 1 else 0.0
    if lam1 < 1e-12 or (lam1 > 0 and (lam1 - lam2) / (lam1 + 1e-12) < min_eigval_ratio):
        return points.astype(np.float32), np.eye(3, dtype=np.float32), mu.astype(np.float32)

    v_main = evecs[:, order[0]].astype(np.float64).reshape(3)
    v_main /= np.linalg.norm(v_main) + 1e-12

    ax = target_axis.lower()
    if ax == 'x':
        target = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    elif ax == 'y':
        target = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    else:
        target = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    if np.dot(v_main, target) < 0:
        v_main = -v_main

    B = _orthonormal_frame_from_axis(v_main)
    Tm = _orthonormal_frame_from_axis(target)
    R64 = Tm @ B.T
    if np.linalg.det(R64) < 0:
        Tm[:, 1] *= -1.0
        R64 = Tm @ B.T
    R = R64.astype(np.float32)

    X = pts - mu
    aligned = ((R64 @ X.T).T + mu).astype(np.float32)
    return aligned, R, mu.astype(np.float32)


def apply_pca_rigid(points: np.ndarray, mu: np.ndarray, R: np.ndarray) -> np.ndarray:
    """与 pca_align_principal_to_axis 中相同的 (p - mu) @ R.T + mu。"""
    mu64 = np.asarray(mu, dtype=np.float64).reshape(1, 3)
    R64 = np.asarray(R, dtype=np.float64).reshape(3, 3)
    X = np.asarray(points, dtype=np.float64) - mu64
    return ((R64 @ X.T).T + mu64).astype(np.float32)


def draw_overlay_same_origin(p_norm: np.ndarray, p_gt_sync: np.ndarray, window_name: str):
    """同一原点、无平移偏移：红=残缺，绿=真值（浅绿近似半透明叠看）。"""
    pcd_n = o3d.geometry.PointCloud()
    pcd_n.points = o3d.utility.Vector3dVector(np.asarray(p_norm, dtype=np.float64))
    pcd_n.paint_uniform_color([1.0, 0.0, 0.0])

    pcd_g = o3d.geometry.PointCloud()
    pcd_g.points = o3d.utility.Vector3dVector(np.asarray(p_gt_sync, dtype=np.float64))
    pcd_g.paint_uniform_color([0.25, 0.85, 0.35])

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)
    vis.add_geometry(pcd_n)
    vis.add_geometry(pcd_g)
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.05, 0.05, 0.08])
    vis.run()
    vis.destroy_window()


def process_single_file(args):
    import open3d as o3d
    import os
    import numpy as np
    import tempfile

    (
        file_path,
        cat,
        split,
        output_path,
        num_points,
        trans_scale_min,
        trans_scale_max,
        use_oriented_bbox,
        show_vis,
        rot_max_deg,
        pca_axis,
        use_pca,
    ) = args

    tmp_path = None

    try:
        is_repaired = False
        with open(file_path, 'rb') as f:
            raw_content = f.read().decode('utf-8', errors='ignore')

        first_line = raw_content.split('\n', 1)[0].strip()
        final_content = raw_content
        if first_line.startswith('OFF') and len(first_line) > 3 and first_line[3].isdigit():
            final_content = 'OFF\n' + raw_content[3:].lstrip()
            is_repaired = True

        _, tmp_path = tempfile.mkstemp(suffix='.off')

        with open(tmp_path, 'w', encoding='utf-8') as f_tmp:
            f_tmp.write(final_content)

        mesh = o3d.io.read_triangle_mesh(tmp_path)

        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            tmp_path = None

        if not mesh.has_triangles():
            return False, file_path, False

        pcd = mesh.sample_points_uniformly(number_of_points=num_points)
        gt_points, gt_normals = normalize_point_cloud(pcd)

        # 场景尺度：GT AABB 最大边长，用于平移系数 0.5~2.0 的物理可比缩放
        gt_min = gt_points.min(axis=0)
        gt_max = gt_points.max(axis=0)
        extent = float(np.max(gt_max - gt_min))

        base_name = os.path.basename(file_path).replace('.off', '')
        num_views = 8

        idx_gt = np.random.choice(len(gt_points), 2048, replace=False)
        gt_save = gt_points[idx_gt]

        for view_idx in range(num_views):
            r = 3.0
            # 视角约定：
            # - v0 固定为“俯拍”：相机在 +Z 正上方（2.5D / HPR 的 top-down 观测）
            # - v1~v7 维持随机视角（增强多视角退化）
            if view_idx == 0:
                camera_pos = np.array([0.0, 0.0, r], dtype=np.float32)
            else:
                theta = np.random.uniform(0, 2 * np.pi)
                phi = np.random.uniform(0, np.pi)
                camera_pos = np.array([
                    r * np.sin(phi) * np.cos(theta),
                    r * np.sin(phi) * np.sin(theta),
                    r * np.cos(phi),
                ], dtype=np.float32)

            pcd_normalized = o3d.geometry.PointCloud()
            pcd_normalized.points = o3d.utility.Vector3dVector(gt_points)
            _, pt_map = pcd_normalized.hidden_point_removal(camera_pos, 100)

            visible_points = gt_points[pt_map]
            visible_normals = gt_normals[pt_map]

            expected_rate = 0.2

            if np.random.rand() > 0.5:
                corrupted, _ = apply_depth_dropout(
                    visible_points,
                    visible_normals,
                    camera_pos=camera_pos,
                    missing_rate=expected_rate,
                )
            else:
                theta_l = np.random.uniform(0, 2 * np.pi)
                phi_l = np.random.uniform(0, np.pi)
                light_dir = np.array([
                    np.sin(phi_l) * np.cos(theta_l),
                    np.sin(phi_l) * np.sin(theta_l),
                    np.cos(phi_l),
                ])
                corrupted, _ = apply_specular_dropout(
                    visible_points,
                    visible_normals,
                    camera_pos=camera_pos,
                    light_dir=light_dir,
                    missing_rate=expected_rate,
                )

            noise = np.random.normal(0, 0.002, corrupted.shape).astype(np.float32)
            corrupted = corrupted + noise

            stem = f'{base_name}_v{view_idx}'
            file_name = f'{stem}.npy'
            meta_name = f'{stem}.npz'

            n_corrupted = corrupted.shape[0]
            if n_corrupted >= 2048:
                idx_in = np.random.choice(n_corrupted, 2048, replace=False)
            else:
                idx_in = np.random.choice(n_corrupted, 2048, replace=True)
            p_obs = corrupted[idx_in].astype(np.float32)

            # ── 随机 6DoF：小角度旋转 + 平移（粗对齐后残余）──────────────────────
            R_rand = random_rotation_matrix(rot_max_deg)
            t_rand = random_translation_vector(trans_scale_min, trans_scale_max, extent)
            p_obs_w = apply_random_se3(p_obs, R_rand, t_rand)
            p_gt_w = apply_random_se3(gt_save, R_rand, t_rand)

            # ── 仅按残缺观测包围盒中心归一化（禁止质心）──────────────────────────
            p_norm, c_bbox = normalize_by_bbox(p_obs_w, use_oriented=use_oriented_bbox)

            # 与 P_norm 同一坐标系：真值经相同刚性变换后再减同一 C_bbox
            p_gt_sync = (p_gt_w.astype(np.float32) - c_bbox).astype(np.float32)

            # ── PCA：主轴对齐（协方差来自 Open3D；P_gt_sync 同步同一 μ,R）────────
            if use_pca:
                p_norm_out, R_pca, mu_pca = pca_align_principal_to_axis(
                    p_norm, target_axis=pca_axis,
                )
                p_gt_out = apply_pca_rigid(p_gt_sync, mu_pca, R_pca)
            else:
                p_norm_out, p_gt_out = p_norm, p_gt_sync
                R_pca, mu_pca = np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)

            c_norm = aabb_center(p_norm_out)
            c_gt = aabb_center(p_gt_out)
            offset_err = float(np.linalg.norm(c_norm - c_gt))

            pca_tag = '+PCA' if use_pca else ''
            print(
                f'[align-offset] {file_name}  '
                f'||C_aabb(P_out)-C_aabb(P_gt_out)||={offset_err:.6f}  '
                f'(bbox 中心化后{pca_tag}；评估残缺相对真值流形偏移)'
            )

            split_root = os.path.join(output_path, split)
            np.save(os.path.join(split_root, 'obs', file_name), p_obs_w.astype(np.float32))
            np.save(os.path.join(split_root, 'input', file_name), p_norm_out.astype(np.float32))
            np.save(os.path.join(split_root, 'gt', file_name), p_gt_out.astype(np.float32))
            np.savez(
                os.path.join(split_root, 'meta', meta_name),
                R_rand=np.asarray(R_rand, dtype=np.float32),
                t_rand=np.asarray(t_rand, dtype=np.float32),
                C_bbox=np.asarray(c_bbox, dtype=np.float32),
                R_pca=np.asarray(R_pca, dtype=np.float32),
                mu_pca=np.asarray(mu_pca, dtype=np.float32),
            )

            if show_vis and view_idx == 0:
                print(
                    f'可视化(同原点): 红=P_out 绿=P_gt_out; '
                    f'||ΔC_aabb||={offset_err:.6f}'
                )
                draw_overlay_same_origin(
                    p_norm_out,
                    p_gt_out,
                    window_name='同坐标系: P_out(红) vs P_gt_out(绿)',
                )

        return True, file_path, is_repaired

    except Exception as e:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        print(f'处理 {file_path} 出错: {e}')
        return False, file_path, False


def main():
    parser = argparse.ArgumentParser(
        description='带随机位姿 + 包围盒归一化的预处理（01_preprocessing 扩展）',
    )
    parser.add_argument(
        '--raw-dir',
        default='/home/csy/SnowflakeNet_FPFH_ICP/data/raw/ModelNet40',
        help='ModelNet40 根目录',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='输出目录，默认 <项目>/data/processed_with_removal',
    )
    parser.add_argument(
        '--trans-scale-min',
        type=float,
        default=0.5,
        help='平移模长系数下限，实际模长 ∈ [min,max]×extent（extent 为 GT AABB 最大边长）',
    )
    parser.add_argument(
        '--trans-scale-max',
        type=float,
        default=2.0,
        help='平移模长系数上限',
    )
    parser.add_argument(
        '--oriented-bbox',
        action='store_true',
        help='normalize_by_bbox 使用 OBB；默认 AABB',
    )
    parser.add_argument(
        '--rot-max-deg',
        type=float,
        default=15.0,
        help='绕 X/Y/Z 各轴随机旋转角上界（度），区间为 [-v,+v]',
    )
    parser.add_argument(
        '--pca-axis',
        choices=('x', 'y', 'z'),
        default='z',
        help='PCA 后将最大特征值主轴对齐到该坐标轴正方向（ModelNet 常用 Z 向上）',
    )
    parser.add_argument(
        '--no-pca',
        action='store_true',
        help='跳过 PCA 主轴对齐（仅 bbox 归一化）',
    )
    parser.add_argument(
        '--vis',
        action='store_true',
        help='弹出 Open3D（仅处理每个模型的第一个视角时显示；建议配合单任务或少量任务）',
    )
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='并行进程数；--vis 时强制为 1',
    )
    args = parser.parse_args()

    RAW_DIR = args.raw_dir
    current_script_path = os.path.dirname(os.path.abspath(__file__))
    if args.output_dir:
        OUTPUT_DIR = os.path.abspath(args.output_dir)
    else:
        OUTPUT_DIR = os.path.abspath(
            os.path.join(current_script_path, '../data/processed_with_removal'),
        )

    if not os.path.exists(RAW_DIR):
        print(f'原始数据集路径不存在: {RAW_DIR}')
        return

    tasks = []
    print(f'正在扫描数据集: {RAW_DIR} ...')
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if f.endswith('.off'):
                file_path = os.path.join(root, f)
                parts = file_path.replace('\\', '/').split('/')
                try:
                    idx = parts.index('ModelNet40')
                    cat = parts[idx + 1]
                    split = parts[idx + 2]

                    for sub in ('gt', 'input', 'obs', 'meta'):
                        os.makedirs(os.path.join(OUTPUT_DIR, split, sub), exist_ok=True)

                    tasks.append(
                        (
                            file_path,
                            cat,
                            split,
                            OUTPUT_DIR,
                            8192,
                            args.trans_scale_min,
                            args.trans_scale_max,
                            args.oriented_bbox,
                            args.vis,
                            args.rot_max_deg,
                            args.pca_axis,
                            not args.no_pca,
                        ),
                    )
                except (ValueError, IndexError):
                    continue

    print(f'共发现 {len(tasks)} 个待处理模型。')

    max_workers = 1 if args.vis else max(1, args.max_workers)
    print(f'开始批处理 (workers={max_workers})，输出: {OUTPUT_DIR}')

    if max_workers == 1:
        results = [process_single_file(t) for t in tqdm(tasks)]
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(executor.map(process_single_file, tasks), total=len(tasks)))

    success_count = sum(1 for r in results if r[0])
    repaired_count = sum(1 for r in results if r[2])
    print('\n' + '=' * 40)
    print('处理完成')
    print(f'成功: {success_count} / {len(tasks)}')
    print(f'修复 Header: {repaired_count}')
    print(f'输出目录: {OUTPUT_DIR}')
    print('=' * 40)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
