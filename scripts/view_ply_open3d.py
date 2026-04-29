#!/usr/bin/env python3
"""
本地查看导出的 .ply（Open3D 窗口）。

用法::

  python scripts/view_ply_open3d.py path/to/a.ply
  python scripts/view_ply_open3d.py *.ply

多个文件时逐个弹窗（关闭当前窗口后显示下一个）。

依赖: pip install open3d
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> int:
    ap = argparse.ArgumentParser(description="用 Open3D 打开 ply 点云")
    ap.add_argument(
        "ply_paths",
        nargs="+",
        help="一个或多个 .ply 路径",
    )
    args = ap.parse_args()

    try:
        import open3d as o3d
    except ImportError:
        print("请先安装: pip install open3d", file=sys.stderr)
        return 1

    for path in args.ply_paths:
        path = os.path.abspath(os.path.expanduser(path))
        if not os.path.isfile(path):
            print(f"跳过（不存在）: {path}", file=sys.stderr)
            continue
        pcd = o3d.io.read_point_cloud(path)
        if not pcd.has_points():
            print(f"跳过（空）: {path}", file=sys.stderr)
            continue
        o3d.visualization.draw_geometries(
            [pcd],
            window_name=os.path.basename(path)[:80],
            width=1280,
            height=720,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
