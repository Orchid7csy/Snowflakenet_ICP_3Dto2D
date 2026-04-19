"""
scripts/11_compute_base_poses.py

为 ModelNet40 原始 mesh 计算「最大可视特征面」基础位姿，结果以 JSON 形式
单独保存（不修改现有预处理 / meta / dataloader）。

设计要点：
- 键名 = 原始 mesh stem（与 scripts/01_preprocess_modelnet40.py 里 `base_name` 完全一致，
  例如 airplane_0627）。预处理生成的样本 stem 形如
      <base_name>[_aug{aug_id:02d}]_v{vi}
  推理阶段把 _aug##、_v# 后缀剥掉即可一一映射。
- 与现有 meta 的关系：本文件只存「每个 mesh 一份」的基础规范位姿 R_align，
  使候选方向 V_max 旋到 +Z；不会替代 meta 里逐样本的 R_pca / scale / C_bbox。
- 仅依赖 numpy / open3d / tqdm。可选 multiprocessing。

输出格式 (base_poses.json)：
{
  "version": 1,
  "params": {...},                          # 复现所需参数
  "poses": {
      "airplane_0627": [[..4x4..]],         # R_align 嵌入到 4x4 齐次矩阵
      ...
  },
  "meta": {
      "airplane_0627": {
          "category": "airplane",
          "split": "train",
          "n_visible": 1843,
          "best_dir_idx": 27,
          "V_max": [vx, vy, vz]
      },
      ...
  }
}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d
from tqdm import tqdm


# ═══════════════════════════ 数学：方向采样与旋转 ═══════════════════════════


def fibonacci_sphere_directions(n: int) -> np.ndarray:
    """生成 n 个近似均匀分布的单位方向向量, shape (n, 3), float64."""
    golden = (1.0 + np.sqrt(5.0)) / 2.0
    i = np.arange(n, dtype=np.float64)
    phi = np.arccos(1.0 - 2.0 * (i + 0.5) / n)
    theta = 2.0 * np.pi * i / golden
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    dirs = np.stack([x, y, z], axis=1)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-15
    return dirs


def rotation_align_v_to_z(v: np.ndarray) -> np.ndarray:
    """返回 3x3 旋转矩阵 R, 使得 R @ v_hat == [0, 0, 1] (右手, det=+1)."""
    a = np.asarray(v, dtype=np.float64).reshape(3)
    na = np.linalg.norm(a)
    if na < 1e-12:
        return np.eye(3, dtype=np.float64)
    a = a / na
    b = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    c = float(np.dot(a, b))
    if c > 1.0 - 1e-12:
        return np.eye(3, dtype=np.float64)
    if c < -1.0 + 1e-12:
        return np.array(
            [[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], dtype=np.float64
        )
    axis = np.cross(a, b)
    axis /= np.linalg.norm(axis) + 1e-15
    s = np.sqrt(max(0.0, 1.0 - c * c))
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + K + K @ K * ((1.0 - c) / (s * s + 1e-15))


def make_homogeneous(R: np.ndarray, t: Optional[np.ndarray] = None) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


# ═══════════════════════════ 几何 I/O ═══════════════════════════


def _read_off_mesh(file_path: str) -> o3d.geometry.TriangleMesh:
    """与 scripts/01_preprocess_modelnet40._read_off_mesh 等价，处理 'OFF<digits>' 头。"""
    with open(file_path, "rb") as f:
        raw = f.read().decode("utf-8", errors="ignore")
    first_line = raw.split("\n", 1)[0].strip()
    content = raw
    if first_line.startswith("OFF") and len(first_line) > 3 and first_line[3].isdigit():
        content = "OFF\n" + raw[3:].lstrip()
    _, tmp = tempfile.mkstemp(suffix=".off")
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
        mesh = o3d.io.read_triangle_mesh(tmp)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return mesh


def load_points_from_path(path: str, mesh_num_samples: int) -> np.ndarray:
    """读 mesh / 点云 → (N, 3) float64，并做与训练一致的『中心化 + 单位球缩放』。"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".off":
        mesh = _read_off_mesh(path)
        if mesh.has_triangles():
            pcd = mesh.sample_points_uniformly(number_of_points=mesh_num_samples)
            pts = np.asarray(pcd.points, dtype=np.float64)
        else:
            pts = np.asarray(mesh.vertices, dtype=np.float64)
    elif ext in (".obj", ".stl", ".ply"):
        mesh = o3d.io.read_triangle_mesh(path)
        if mesh.has_triangles():
            pcd = mesh.sample_points_uniformly(number_of_points=mesh_num_samples)
            pts = np.asarray(pcd.points, dtype=np.float64)
        else:
            pcd = o3d.io.read_point_cloud(path)
            pts = np.asarray(pcd.points, dtype=np.float64)
    elif ext == ".pcd":
        pcd = o3d.io.read_point_cloud(path)
        pts = np.asarray(pcd.points, dtype=np.float64)
    else:
        raise ValueError(f"unsupported extension: {ext}")

    if pts.size == 0:
        return pts.astype(np.float64)

    # 与 _normalize_point_cloud 一致：减质心 + 缩放到单位球（max radius -> 1）
    pts = pts - pts.mean(axis=0, keepdims=True)
    r = float(np.linalg.norm(pts, axis=1).max())
    if r > 1e-12:
        pts = pts / r
    return pts.astype(np.float64)


# ═══════════════════════════ HPR 视点搜索 ═══════════════════════════


def compute_camera_distance(points: np.ndarray, distance_factor: float) -> float:
    """相机距离 = factor * AABB 对角线长度 (与现有预处理 hpr_sphere_r 同口径)."""
    mn, mx = points.min(axis=0), points.max(axis=0)
    diag = float(np.linalg.norm(mx - mn))
    return max(diag, 1e-8) * float(distance_factor)


def compute_hpr_radius(cam_dist: float, radius_factor: float) -> float:
    """Open3D HPR 推荐: radius >> 相机到最远点的距离。
    取 radius = radius_factor * cam_dist (factor 默认 100)."""
    return float(radius_factor) * float(cam_dist)


def best_view_hpr(
    points: np.ndarray,
    directions: np.ndarray,
    distance_factor: float,
    radius_factor: float,
) -> Tuple[np.ndarray, int, int]:
    """对每个候选方向跑 HPR, 返回 (V_max(单位向量), 最大可见点数, 该方向索引)."""
    if points.shape[0] == 0:
        return directions[0].astype(np.float64), 0, 0

    cam_dist = compute_camera_distance(points, distance_factor)
    hpr_radius = compute_hpr_radius(cam_dist, radius_factor)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    best_n = -1
    best_i = 0
    for i, d in enumerate(directions):
        cam = d * cam_dist
        try:
            _, pt_map = pcd.hidden_point_removal(cam.astype(np.float64), hpr_radius)
        except RuntimeError:
            continue
        n_vis = len(pt_map)
        if n_vis > best_n:
            best_n = n_vis
            best_i = i

    V_max = directions[best_i].astype(np.float64)
    V_max /= np.linalg.norm(V_max) + 1e-15
    return V_max, int(best_n), int(best_i)


# ═══════════════════════════ 数据集扫描 ═══════════════════════════


def _scan_modelnet40(raw_dir: str, exts: Tuple[str, ...]) -> List[Tuple[str, str, str, str]]:
    """
    返回 [(stem, file_path, category, split), ...]，
    与 scripts/01_preprocess_modelnet40._scan_dataset 命名一致：
    stem = os.path.basename(file_path) 去后缀，用作 base_poses.json 的键。
    """
    entries: List[Tuple[str, str, str, str]] = []
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if not f.lower().endswith(exts):
                continue
            fp = os.path.join(root, f)
            stem = os.path.splitext(f)[0]
            parts = fp.replace("\\", "/").split("/")
            cat, split = "", ""
            try:
                idx = parts.index("ModelNet40")
                cat = parts[idx + 1]
                split = parts[idx + 2]
            except (ValueError, IndexError):
                pass
            entries.append((stem, fp, cat, split))
    entries.sort(key=lambda x: x[0])
    return entries


# ═══════════════════════════ Worker ═══════════════════════════


def _process_one(job: dict) -> Tuple[str, dict, dict]:
    stem = job["stem"]
    pts = load_points_from_path(job["path"], job["mesh_num_samples"])
    dirs = fibonacci_sphere_directions(job["n_directions"])
    V_max, n_vis, idx = best_view_hpr(
        pts,
        dirs,
        distance_factor=job["distance_factor"],
        radius_factor=job["radius_factor"],
    )
    R = rotation_align_v_to_z(V_max)
    T4 = make_homogeneous(R)

    pose_entry = T4.tolist()
    meta_entry = {
        "category": job["category"],
        "split": job["split"],
        "n_visible": int(n_vis),
        "best_dir_idx": int(idx),
        "V_max": [float(V_max[0]), float(V_max[1]), float(V_max[2])],
        "source_path": job["path"],
    }
    return stem, pose_entry, meta_entry


# ═══════════════════════════ Helper for inference ═══════════════════════════


def base_key_from_sample_stem(sample_stem: str) -> str:
    """
    把 01_preprocess_modelnet40.py 产生的样本 stem 还原成原始 mesh 的 base key。
    规则：去掉末尾的 _v{int} 与可选的 _aug{int}，例如
        airplane_0627_aug03_v5 -> airplane_0627
        airplane_0627_v0       -> airplane_0627
    """
    s = sample_stem
    # 末尾 _v\d+
    if "_v" in s:
        head, tail = s.rsplit("_v", 1)
        if tail.isdigit():
            s = head
    # 末尾 _aug\d+
    if "_aug" in s:
        head, tail = s.rsplit("_aug", 1)
        if tail.isdigit():
            s = head
    return s


# ═══════════════════════════ Main ═══════════════════════════


def main():
    ap = argparse.ArgumentParser(
        description="ModelNet40 best-view (HPR) base poses generator (独立脚本, 不改动现有预处理)."
    )
    ap.add_argument(
        "--raw-dir",
        default="/home/csy/SnowflakeNet_FPFH_ICP/data/raw/ModelNet40",
        help="原始 ModelNet40 根目录（包含 <category>/<split>/*.off）",
    )
    ap.add_argument(
        "--out-json",
        default="/home/csy/SnowflakeNet_FPFH_ICP/data/processed/modelnet40/base_poses.json",
        help="输出 JSON 路径",
    )
    ap.add_argument("--ext", default=".off", help="逗号分隔扩展名, 如 .off,.ply")
    ap.add_argument("--n-directions", type=int, default=100, help="Fibonacci 球面采样方向数")
    ap.add_argument(
        "--distance-factor",
        type=float,
        default=3.0,
        help="相机距离 = factor * AABB 对角线长度 (与 01_preprocess_modelnet40 的 hpr_sphere_r 同口径)",
    )
    ap.add_argument(
        "--radius-factor",
        type=float,
        default=100.0,
        help="HPR radius = radius_factor * camera_distance (Open3D 推荐 radius >> 相机到最远点距离)",
    )
    ap.add_argument("--mesh-num-samples", type=int, default=16384, help="mesh 均匀采样点数")
    ap.add_argument("--workers", type=int, default=1, help=">1 时使用多进程加速")
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只处理前 N 个样本（调试用，0 表示全部）",
    )
    args = ap.parse_args()

    if not os.path.isdir(args.raw_dir):
        print(f"[ERR] raw-dir 不存在: {args.raw_dir}", file=sys.stderr)
        sys.exit(1)

    exts = tuple(e.strip().lower() for e in args.ext.split(",") if e.strip())
    entries = _scan_modelnet40(args.raw_dir, exts)
    if args.limit > 0:
        entries = entries[: args.limit]
    if not entries:
        print("[ERR] 未找到任何 mesh 文件", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] 共 {len(entries)} 个 mesh, 方向数 = {args.n_directions}, workers = {args.workers}")

    jobs = [
        {
            "stem": stem,
            "path": fp,
            "category": cat,
            "split": split,
            "n_directions": args.n_directions,
            "distance_factor": args.distance_factor,
            "radius_factor": args.radius_factor,
            "mesh_num_samples": args.mesh_num_samples,
        }
        for (stem, fp, cat, split) in entries
    ]

    poses: Dict[str, List[List[float]]] = {}
    metas: Dict[str, dict] = {}

    if args.workers and args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_process_one, j): j["stem"] for j in jobs}
            for fut in tqdm(as_completed(futs), total=len(futs), desc="HPR best-view"):
                try:
                    stem, pose, meta = fut.result()
                except Exception as e:  # noqa: BLE001
                    print(f"[WARN] 处理 {futs[fut]} 失败: {e}", file=sys.stderr)
                    continue
                poses[stem] = pose
                metas[stem] = meta
    else:
        for j in tqdm(jobs, desc="HPR best-view"):
            try:
                stem, pose, meta = _process_one(j)
            except Exception as e:  # noqa: BLE001
                print(f"[WARN] 处理 {j['stem']} 失败: {e}", file=sys.stderr)
                continue
            poses[stem] = pose
            metas[stem] = meta

    # 与扫描顺序保持一致（dict 在 py3.7+ 有序）
    poses_sorted = {k: poses[k] for k in sorted(poses)}
    metas_sorted = {k: metas[k] for k in sorted(metas)}

    payload = {
        "version": 1,
        "params": {
            "n_directions": args.n_directions,
            "distance_factor": args.distance_factor,
            "radius_factor": args.radius_factor,
            "mesh_num_samples": args.mesh_num_samples,
            "normalization": "center+unit_sphere",
            "rotation_convention": "R @ V_max == [0,0,1]",
            "key_naming": "raw mesh stem; for processed sample stem use base_key_from_sample_stem()",
        },
        "poses": poses_sorted,
        "meta": metas_sorted,
    }

    out_dir = os.path.dirname(os.path.abspath(args.out_json))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[OK] wrote {args.out_json}  entries={len(poses_sorted)}")


if __name__ == "__main__":
    main()
