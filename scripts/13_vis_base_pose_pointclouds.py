"""
scripts/13_vis_base_pose_pointclouds.py

base_poses.json 只存 4x4 位姿与 meta，不存点云。本脚本：
  从 meta.source_path 读 mesh → 与 11_compute_base_poses 相同方式归一化采样
  → 左乘 R^T（行向量点云 p @ R.T）对齐到 canonical（原 V_max → +Z）
  → Open3D 可视化。

示例：
  python scripts/13_vis_base_pose_pointclouds.py --stem airplane_0627,chair_0891
  python scripts/13_vis_base_pose_pointclouds.py -n 3 --seed 0
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import numpy as np
import open3d as o3d

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))


def _load_bp_mod():
    spec = importlib.util.spec_from_file_location(
        "compute_base_poses", os.path.join(_SCRIPT_DIR, "11_compute_base_poses.py")
    )
    if spec is None or spec.loader is None:
        raise ImportError("无法加载 11_compute_base_poses.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser(description="可视化 base_pose 对齐后的点云（不落盘）")
    ap.add_argument(
        "--base-json",
        default=os.path.join(_PROJECT_ROOT, "data", "processed", "modelnet40", "base_poses.json"),
    )
    ap.add_argument("--stem", default="", help="逗号分隔的 mesh stem；为空则用 -n 随机抽")
    ap.add_argument("-n", "--num-random", type=int, default=0, help="从 JSON 中随机抽几个（与 --stem 互斥优先 stem）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mesh-samples", type=int, default=0, help="0 表示用 JSON params.mesh_num_samples 或 16384")
    ap.add_argument(
        "--offset",
        type=float,
        default=2.5,
        help="多模型同屏时沿 X 平移间距（0 则叠在原点）",
    )
    ap.add_argument(
        "--raw-dir",
        default=os.path.join(_PROJECT_ROOT, "data", "raw", "ModelNet40"),
        help="meta 无 source_path 时拼 <raw-dir>/<category>/<split>/<stem>.off",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.base_json):
        print(f"找不到 {args.base_json}", file=sys.stderr)
        sys.exit(1)

    with open(args.base_json, encoding="utf-8") as f:
        data = json.load(f)

    poses = data.get("poses") or {}
    metas = data.get("meta") or {}
    params = data.get("params") or {}
    n_samp = args.mesh_samples or int(params.get("mesh_num_samples", 16384))

    if args.stem:
        stems = [s.strip() for s in args.stem.split(",") if s.strip()]
    else:
        all_keys = sorted(metas.keys())
        if args.num_random <= 0:
            print("请指定 --stem 或 -n > 0", file=sys.stderr)
            sys.exit(1)
        rng = np.random.default_rng(args.seed)
        stems = list(rng.choice(all_keys, size=min(args.num_random, len(all_keys)), replace=False))

    bp = _load_bp_mod()
    geoms = []
    palette = [
        [0.2, 0.45, 0.95],
        [0.95, 0.35, 0.15],
        [0.25, 0.8, 0.35],
        [0.85, 0.25, 0.75],
        [0.9, 0.85, 0.2],
    ]

    for i, stem in enumerate(stems):
        if stem not in poses or stem not in metas:
            print(f"[skip] 无此键: {stem}", file=sys.stderr)
            continue
        m = metas[stem]
        path = m.get("source_path")
        if not path or not os.path.isfile(path):
            cat, sp = m.get("category", ""), m.get("split", "")
            if cat and sp:
                cand = os.path.join(args.raw_dir, str(cat), str(sp), f"{stem}.off")
                if os.path.isfile(cand):
                    path = cand
        if not path or not os.path.isfile(path):
            print(f"[skip] 无有效 mesh 路径: {stem}", file=sys.stderr)
            continue

        pts = bp.load_points_from_path(path, n_samp)
        if pts.shape[0] == 0:
            print(f"[skip] 空点云: {stem}", file=sys.stderr)
            continue

        T = np.asarray(poses[stem], dtype=np.float64).reshape(4, 4)
        R = T[:3, :3]
        aligned = (pts @ R.T).astype(np.float64)

        dx = i * args.offset if args.offset else 0.0
        aligned = aligned + np.array([dx, 0.0, 0.0], dtype=np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(aligned)
        pcd.paint_uniform_color(palette[i % len(palette)])

        geoms.append(pcd)
        print(f"{stem}: N={pts.shape[0]}  path={path}")

    if not geoms:
        print("没有可显示的几何体", file=sys.stderr)
        sys.exit(1)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.25, origin=[0, 0, 0])
    geoms.append(origin)

    o3d.visualization.draw_geometries(
        geoms,
        window_name="base_pose 对齐点云 (+Z 为原最佳观察方向)",
        width=1280,
        height=720,
    )


if __name__ == "__main__":
    main()
