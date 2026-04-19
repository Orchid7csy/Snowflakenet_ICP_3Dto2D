"""
scripts/12_validate_base_poses_vis.py

验证 base_poses.json 中记录的最佳视角：在相同归一化与 HPR 参数下重算可见点数，
与 JSON meta 中的 n_visible 对比；可选 Open3D 可视化（全点云 vs HPR 保留点 + 相机位置）。

用法示例：
  # 批量统计（无窗口）
  python scripts/12_validate_base_poses_vis.py \\
    --base-json data/processed/modelnet40/base_poses.json --summary-only

  # 单个模型 + 可视化
  python scripts/12_validate_base_poses_vis.py \\
    --base-json data/processed/modelnet40/base_poses.json \\
    --stem airplane_0627 --vis
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import open3d as o3d

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))


def _load_base_poses_module():
    path = os.path.join(_SCRIPT_DIR, "11_compute_base_poses.py")
    spec = importlib.util.spec_from_file_location("compute_base_poses", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hpr_visible_count(
    points: np.ndarray, camera_pos: np.ndarray, hpr_radius: float
) -> Tuple[int, List[int]]:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    _, pt_map = pcd.hidden_point_removal(
        camera_pos.astype(np.float64), float(hpr_radius)
    )
    idx = list(pt_map)
    return len(idx), idx


def _make_vis_geometries(
    points: np.ndarray,
    visible_idx: List[int],
    camera_pos: np.ndarray,
    show_camera_marker: bool,
) -> List[o3d.geometry.Geometry]:
    vis_set = set(visible_idx)
    hidden_idx = [i for i in range(points.shape[0]) if i not in vis_set]

    pcd_hidden = o3d.geometry.PointCloud()
    if hidden_idx:
        pcd_hidden.points = o3d.utility.Vector3dVector(points[hidden_idx].astype(np.float64))
        pcd_hidden.paint_uniform_color([0.35, 0.55, 0.35])

    pcd_vis = o3d.geometry.PointCloud()
    pcd_vis.points = o3d.utility.Vector3dVector(points[visible_idx].astype(np.float64))
    pcd_vis.paint_uniform_color([0.95, 0.25, 0.15])

    geoms: List[o3d.geometry.Geometry] = []
    if hidden_idx:
        geoms.append(pcd_hidden)
    geoms.append(pcd_vis)

    if show_camera_marker:
        # 相机标记尺寸按点云尺度自适应；过远时也不至于撑爆视野（仍可拖拽看到）。
        mn, mx = points.min(axis=0), points.max(axis=0)
        diag = float(np.linalg.norm(mx - mn))
        ball_r = max(diag * 0.02, 1e-3)
        frame_size = max(diag * 0.1, 1e-3)
        cam_ball = o3d.geometry.TriangleMesh.create_sphere(radius=ball_r)
        cam_ball.translate(camera_pos)
        cam_ball.paint_uniform_color([0.1, 0.4, 0.95])
        cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
            size=frame_size, origin=[0.0, 0.0, 0.0]
        )
        cam_frame.translate(camera_pos)
        geoms.extend([cam_ball, cam_frame])

    return geoms


def _draw_centered_on_cloud(
    geoms: List[o3d.geometry.Geometry],
    points: np.ndarray,
    V_max: np.ndarray,
    window_name: str,
) -> None:
    """显式把视角对准点云包围盒中心，避免远处相机标记把点云挤成一个像素。"""
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1280, height=720)
    for g in geoms:
        vis.add_geometry(g)

    center = (points.min(axis=0) + points.max(axis=0)) * 0.5
    diag = float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))
    front = np.asarray(V_max, dtype=np.float64).reshape(3)
    front /= np.linalg.norm(front) + 1e-15

    # 选个与 front 不平行的 up
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(up, front)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])

    ctr = vis.get_view_control()
    ctr.set_lookat(center.tolist())
    ctr.set_front(front.tolist())
    ctr.set_up(up.tolist())
    ctr.set_zoom(0.7)

    opt = vis.get_render_option()
    opt.point_size = 3.0
    opt.background_color = np.array([1.0, 1.0, 1.0])

    vis.run()
    vis.destroy_window()


def _resolve_mesh_path(meta: Dict[str, Any], raw_dir: str) -> Optional[str]:
    sp = meta.get("source_path")
    if isinstance(sp, str) and os.path.isfile(sp):
        return sp
    cat = meta.get("category") or ""
    split = meta.get("split") or ""
    stem = meta.get("stem")
    if stem and cat and split and raw_dir:
        cand = os.path.join(raw_dir, str(cat), str(split), f"{stem}.off")
        if os.path.isfile(cand):
            return cand
    return None


def validate_one(
    stem: str,
    meta: Dict[str, Any],
    bp_mod: Any,
    raw_dir: str,
    mesh_num_samples: int,
    distance_factor: float,
    radius_factor: float,
    vis: bool,
    show_camera_marker: bool,
) -> Dict[str, Any]:
    V_max = np.asarray(meta["V_max"], dtype=np.float64).reshape(3)
    n_stored = int(meta.get("n_visible", -1))

    path = _resolve_mesh_path({**meta, "stem": stem}, raw_dir)
    if not path:
        return {
            "stem": stem,
            "ok": False,
            "error": "找不到 mesh 路径（检查 meta.source_path 或 --raw-dir）",
        }

    pts = bp_mod.load_points_from_path(path, mesh_num_samples)
    n_total = int(pts.shape[0])
    if n_total == 0:
        return {"stem": stem, "ok": False, "error": "点云为空"}

    cam_dist = bp_mod.compute_camera_distance(pts, distance_factor)
    hpr_r = bp_mod.compute_hpr_radius(cam_dist, radius_factor)
    cam = (V_max / (np.linalg.norm(V_max) + 1e-15)) * cam_dist

    n_now, idx = _hpr_visible_count(pts, cam, hpr_r)
    ratio = n_now / max(n_total, 1)

    out: Dict[str, Any] = {
        "stem": stem,
        "ok": True,
        "path": path,
        "n_total": n_total,
        "n_visible_stored": n_stored,
        "n_visible_recomputed": n_now,
        "delta": n_now - n_stored,
        "visible_ratio": ratio,
        "camera_distance": cam_dist,
        "hpr_radius": hpr_r,
    }

    if vis:
        geoms = _make_vis_geometries(pts, idx, cam, show_camera_marker)
        title = (
            f"base_pose HPR: {stem} | visible {n_now}/{n_total} "
            f"({100*ratio:.1f}%) | cam_dist={cam_dist:.3f} radius={hpr_r:.1f}"
        )
        _draw_centered_on_cloud(geoms, pts, V_max, title)

    return out


def main():
    ap = argparse.ArgumentParser(description="验证 base_poses.json 的 HPR 可见点数并可视化")
    ap.add_argument(
        "--base-json",
        default=os.path.join(_PROJECT_ROOT, "data", "processed", "modelnet40", "base_poses.json"),
    )
    ap.add_argument(
        "--raw-dir",
        default=os.path.join(_PROJECT_ROOT, "data", "raw", "ModelNet40"),
        help="当 meta 无有效 source_path 时，用于拼接 <cat>/<split>/<stem>.off",
    )
    ap.add_argument("--stem", default="", help="只验证指定 stem（逗号分隔多个）")
    ap.add_argument("--limit", type=int, default=0, help="最多验证前 N 条（0=全部）")
    ap.add_argument("--summary-only", action="store_true", help="只打印汇总，不逐条打印")
    ap.add_argument("--vis", action="store_true", help="Open3D 窗口：绿=遮挡，红=HPR 可见，蓝球=相机")
    ap.add_argument("--mesh-num-samples", type=int, default=0, help="0=使用 JSON params 或 16384")
    ap.add_argument("--distance-factor", type=float, default=0.0, help="0=使用 JSON params")
    ap.add_argument("--radius-factor", type=float, default=0.0, help="0=使用 JSON params (HPR radius = factor * cam_dist)")
    ap.add_argument(
        "--no-camera-marker",
        action="store_true",
        help="可视化时不绘制相机小球/坐标轴 (远相机时画面更干净)",
    )
    args = ap.parse_args()

    if not os.path.isfile(args.base_json):
        print(f"[ERR] 找不到 {args.base_json}", file=sys.stderr)
        sys.exit(1)

    with open(args.base_json, encoding="utf-8") as f:
        data: Dict[str, Any] = json.load(f)

    params = data.get("params") or {}
    mesh_n = args.mesh_num_samples or int(params.get("mesh_num_samples", 16384))
    dist_f = args.distance_factor or float(params.get("distance_factor", 3.0))
    rad_f = args.radius_factor or float(params.get("radius_factor", 100.0))

    metas: Dict[str, Any] = data.get("meta") or {}
    if not metas:
        print("[ERR] JSON 中无 meta 字段", file=sys.stderr)
        sys.exit(1)

    if args.stem:
        keys = [s.strip() for s in args.stem.split(",") if s.strip()]
    else:
        keys = sorted(metas.keys())
    if args.limit > 0:
        keys = keys[: args.limit]

    bp_mod = _load_base_poses_module()

    results: List[Dict[str, Any]] = []
    for k in keys:
        if k not in metas:
            results.append({"stem": k, "ok": False, "error": "meta 中无此键"})
            continue
        r = validate_one(
            k,
            metas[k],
            bp_mod,
            args.raw_dir,
            mesh_n,
            dist_f,
            rad_f,
            vis=args.vis and len(keys) == 1,
            show_camera_marker=not args.no_camera_marker,
        )
        results.append(r)
        if not args.summary_only and r.get("ok"):
            print(
                f"{k}: total={r['n_total']} recomputed={r['n_visible_recomputed']} "
                f"stored={r['n_visible_stored']} delta={r['delta']} "
                f"ratio={r['visible_ratio']*100:.2f}%"
            )
        elif not args.summary_only and not r.get("ok"):
            print(f"{k}: FAIL — {r.get('error', '?')}")

    ok_rows = [r for r in results if r.get("ok")]
    if len(keys) > 1 and args.vis:
        print(
            "[INFO] 多条目模式下已跳过逐条弹窗；请使用 --stem <单个> --vis 查看可视化",
            file=sys.stderr,
        )

    if ok_rows:
        deltas = [abs(r["delta"]) for r in ok_rows]
        print("\n=== Summary ===")
        print(f"checked: {len(results)}  ok: {len(ok_rows)}")
        print(
            f"n_visible: stored vs recomputed — "
            f"exact_match={sum(1 for r in ok_rows if r['delta']==0)}/{len(ok_rows)}"
        )
        if deltas:
            print(f"|delta| max={max(deltas)}  mean={np.mean(deltas):.4f}")
        ratios = [r["visible_ratio"] for r in ok_rows]
        print(
            f"visible_ratio: min={min(ratios)*100:.2f}%  max={max(ratios)*100:.2f}%  "
            f"mean={float(np.mean(ratios))*100:.2f}%"
        )

    bad = [r for r in results if not r.get("ok")]
    if bad:
        print(f"\n[WARN] {len(bad)} 条失败", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
