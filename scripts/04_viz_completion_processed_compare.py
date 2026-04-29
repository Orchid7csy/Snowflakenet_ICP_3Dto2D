#!/usr/bin/env python3
"""
预处理数据集上：官方预训练 vs 微调 —— 与 ``scripts/03_eval_completion.py`` 同一前向逻辑。

- 从 ``{processed-root}/{split}/input|gt`` 读取 paired ``.npy``；
- ``forward_from_npy``（``input_mode`` / ``n-input-points`` / optional ``--use-comp-filter``）与评测脚本一致；
- Open3D 并排：input | official pred | ours pred | GT（可选 ply 导出，无显示器也可用）。

默认微调权重：项目根 ``checkpoints/ckpt-best.pth``（可用 ``--ours-ckpt`` 覆盖）。

**云端（AutoDL 等无显示器）**：默认检测到没有 DISPLAY/WAYLAND，会自动把 ply 写到**当前目录**
``completion_compare_export/``，不尝试弹窗。将整个文件夹拷回笔记本后用 MeshLab / Open3D GUI 打开即可；
若在桌面 Ubuntu/WSL 等有显示器环境下想在云端强制导出而非窗口，仍可加 ``--save-dir``.

**本地有显卡或 CPU**：在项目根执行时不要 ``DISPLAY`` 为空（本地桌面默认即有）；不传 ``--save-dir``
时才弹出 Open3D 并排窗口。

示例::

  # 本地笔记本（克隆仓库 + 数据 + ckpt 后）
  PYTHONPATH=. python scripts/04_viz_completion_processed_compare.py \\
      --processed-root data/processed/PCN_far8_cano_in2048_gt16384 --split test \\
      --sample-index 0

  # AutoDL：自动生成 completion_compare_export/（或可显式指定目录）
  PYTHONPATH=. python scripts/04_viz_completion_processed_compare.py \\
      --processed-root data/processed/PCN_far8_cano_in2048_gt16384 \\
      --sample-index 0 --max-stems 5

  PYTHONPATH=. python scripts/04_viz_completion_processed_compare.py \\
      --processed-root data/processed/PCN_far8_cano_in2048_gt16384 \\
      --save-dir results/viz_pretrain_vs_finetune \\
      --sample-index 0 --max-stems 5
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_SNET_ROOT = os.path.join(_PROJECT_ROOT, "Snet", "SnowflakeNet-main")
_DEFAULT_OFFICIAL_CKPT = os.path.join(
    _SNET_ROOT, "completion", "checkpoints", "ckpt-best-pcn-cd_l1.pth",
)
_DEFAULT_OURS_CKPT = os.path.join(_PROJECT_ROOT, "checkpoints", "ckpt-best.pth")

for p in (_PROJECT_ROOT, _SNET_ROOT, _SCRIPT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.evaluation.cd_l1 import cdl1_times_1e3, list_npy_pairs  # noqa: E402
from src.evaluation.npy_forward import forward_from_npy  # noqa: E402
from src.models.snet_loader import load_snowflakenet  # noqa: E402
from src.utils.io import to_o3d_pcd  # noqa: E402


def _gui_likely_works() -> bool:
    """云端容器通常无 DISPLAY/WAYLAND，Open3D 窗口不可用。"""
    if sys.platform in ("win32", "darwin"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


_DEFAULT_HEADLESS_EXPORT = os.path.join(
    os.getcwd(),
    "completion_compare_export",
)


def _span(pts: np.ndarray) -> float:
    if pts.size == 0:
        return 1.0
    return float(np.ptp(pts, axis=0).max()) or 1.0


def _viz_four_columns(
    *,
    stem: str,
    part: np.ndarray,
    pred_official: np.ndarray,
    pred_ours: np.ndarray,
    gt: np.ndarray | None,
    cd_official_x1e3: float,
    cd_ours_x1e3: float,
    save_dir: str | None,
    processed_root: str,
    split: str,
    official_ckpt: str,
    ours_ckpt: str,
) -> None:
    blobs = [
        (part, (0.95, 0.2, 0.15)),
        (pred_official, (1.0, 0.55, 0.1)),
        (pred_ours, (0.25, 0.85, 0.35)),
    ]
    labels = ["input", "official_pred", "ours_pred"]
    if gt is not None:
        blobs.append((gt, (0.35, 0.45, 0.95)))
        labels.append("gt")

    dx = max(
        _span(part),
        _span(pred_official),
        _span(pred_ours),
        _span(gt) if gt is not None else 1.0,
    )
    sep = dx * 1.6
    x0 = -(len(blobs) - 1) * 0.5 * sep

    geoms = []
    for i, label in enumerate(labels):
        pts, col = blobs[i]
        gx = float(x0 + i * sep)
        geoms.append((label, np.asarray(pts + np.array([gx, 0, 0], dtype=np.float32)), col))

    print("")
    print(f"stem: {stem}")
    print(f"  processed-root: {processed_root}  split={split}")
    print(f"  official: {official_ckpt}")
    print(f"  ours:     {ours_ckpt}")
    print(
        f"  CD-L1×10³ vs GT: official={cd_official_x1e3:.3f}  ours={cd_ours_x1e3:.3f}  "
        f"(Δ={cd_official_x1e3 - cd_ours_x1e3:+.3f}, official−ours)"
    )
    print("  左→右: 红=input, 橙=official, 绿=ours" + (", 蓝=GT" if gt is not None else ""))

    if save_dir:
        import open3d as o3d

        out_root = os.path.abspath(save_dir)
        os.makedirs(out_root, exist_ok=True)
        base = "".join((c if c.isalnum() or c in "-_" else "_" for c in stem))[:180]
        for label, pts, col in geoms:
            path = os.path.join(out_root, f"{base}__processed_compare__{label}.ply")
            o3d.io.write_point_cloud(path, to_o3d_pcd(np.asarray(pts), color=col))
            print(f"  wrote {path}")

        pts_acc = []
        col_acc = []
        for _label, pts, rgb in geoms:
            arr = np.asarray(pts, dtype=np.float64)
            n = arr.shape[0]
            pts_acc.append(arr)
            col_acc.append(np.tile(np.asarray(rgb, dtype=np.float64), (n, 1)))
        merged = o3d.geometry.PointCloud()
        merged.points = o3d.utility.Vector3dVector(np.vstack(pts_acc))
        merged.colors = o3d.utility.Vector3dVector(np.vstack(col_acc))
        merge_path = os.path.join(out_root, f"{base}__processed_compare__merged_side_by_side.ply")
        o3d.io.write_point_cloud(merge_path, merged)
        print(f"  wrote {merge_path}  （单文件并排：红=input, 橙=official, 绿=ours, 蓝=GT）")
        return

    import open3d as o3d

    o3d_geoms = [to_o3d_pcd(pts, color=c) for _name, pts, c in geoms]
    o3d.visualization.draw_geometries(
        o3d_geoms,
        window_name=f"official_vs_ours | CD×10³ off={cd_official_x1e3:.2f} ours={cd_ours_x1e3:.2f} | {stem[:48]}",
        width=1480,
        height=780,
    )


_CLOUD_EPILOG = """
云端（AutoDL，项目在项目根 ~/autodl-tmp/project）示例::

  cd ~/autodl-tmp/project   # 或你的仓库根目录，须含 Snet/、src/

  # 必须指定预处理目录（含 train|test/input 与 gt）；无显示器时自动写出 ply
  PYTHONPATH=. python scripts/04_viz_completion_processed_compare.py \\
      --processed-root data/processed/PCN_far8_cano_in2048_gt16384 \\
      --split test --sample-index 0

  # 指定导出目录（可选）
  PYTHONPATH=. python scripts/04_viz_completion_processed_compare.py \\
      --processed-root data/processed/PCN_far8_cano_in2048_gt16384 \\
      --split test --sample-index 0 --max-stems 3 \\
      --save-dir ~/autodl-tmp/project/completion_compare_export

  # 绝对路径亦可
  PYTHONPATH=. python scripts/04_viz_completion_processed_compare.py \\
      --processed-root /root/autodl-tmp/project/data/processed/PCN_far8_cano_in2048_gt16384 \\
      --split test --sample-index 0

权重默认：官方 ``Snet/.../ckpt-best-pcn-cd_l1.pth``，微调 ``checkpoints/ckpt-best.pth``
（可用 ``--official-ckpt`` / ``--ours-ckpt`` 覆盖）。
"""


def main() -> int:
    ap = argparse.ArgumentParser(
        description="预处理数据上 official vs finetune 补全对比（与 03_eval 同一前向）。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_CLOUD_EPILOG,
    )
    ap.add_argument(
        "--processed-root",
        required=True,
        metavar="DIR",
        help="预处理根目录：须含 {split}/input/*.npy 与 gt/*.npy（与 03_eval 一致）。例：data/processed/PCN_far8_cano_in2048_gt16384",
    )
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--official-ckpt", default=_DEFAULT_OFFICIAL_CKPT)
    ap.add_argument("--ours-ckpt", default=_DEFAULT_OURS_CKPT)
    ap.add_argument("--stem", default=None, help="仅该 stem（不含 .npy）；与区间抽样互斥优先")
    ap.add_argument("--sample-index", type=int, default=0, help="按 stem 排序后的起始下标")
    ap.add_argument(
        "--max-stems",
        type=int,
        default=1,
        help="从 sample-index 起连续处理的样本数（save-dir 下写多组 ply）",
    )
    ap.add_argument(
        "--input-mode",
        choices=("direct", "upsample", "legacy"),
        default="direct",
    )
    ap.add_argument("--use-comp-filter", action="store_true")
    ap.add_argument("--n-input-points", type=int, default=2048)
    ap.add_argument(
        "--save-dir",
        default="",
        help=(
            "写出 ply 到此目录并跳过窗口；若不指定且无显示器则默认写入 ./completion_compare_export/"
        ),
    )
    args = ap.parse_args()

    save_dir_arg = args.save_dir.strip()
    if save_dir_arg:
        resolved_save = os.path.abspath(os.path.expanduser(save_dir_arg))
    elif not _gui_likely_works():
        resolved_save = os.path.abspath(_DEFAULT_HEADLESS_EXPORT)
        print(
            f"[INFO] 当前环境无 DISPLAY/WAYLAND，无法在云端弹 Open3D 窗口；"
            f"自动导出 ply 到: {resolved_save}\n"
            "      将本目录下载到本地后用 MeshLab / Open3D Viewer 打开；"
            "或在本地克隆仓库并放置相同数据后运行本脚本（有桌面则直接弹窗）。",
            file=sys.stderr,
        )
    else:
        resolved_save = ""

    args._resolved_save_dir = resolved_save  # noqa: SLF001

    processed_root = os.path.abspath(args.processed_root)
    pairs = list_npy_pairs(processed_root, args.split)
    if not pairs:
        print(f"未找到成对 .npy：{processed_root}/{args.split}/input|gt", file=sys.stderr)
        return 1

    if args.stem:
        filt = [p for p in pairs if p[0] == args.stem]
        if not filt:
            print(f"stem 不在列表中: {args.stem}", file=sys.stderr)
            return 1
        slice_pairs = filt[: args.max_stems]
    else:
        i0 = args.sample_index
        i1 = i0 + max(args.max_stems, 1)
        if i0 < 0 or i0 >= len(pairs):
            print(f"sample-index 越界: {i0} (len={len(pairs)})", file=sys.stderr)
            return 1
        slice_pairs = pairs[i0:i1]

    official_ckpt = os.path.abspath(os.path.expanduser(args.official_ckpt))
    ours_ckpt = os.path.abspath(os.path.expanduser(args.ours_ckpt))
    for p in (official_ckpt, ours_ckpt):
        if not os.path.isfile(p):
            print(f"权重不存在: {p}", file=sys.stderr)
            return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")

    fwd_kw = dict(
        input_mode=args.input_mode,
        do_comp_filter=args.use_comp_filter,
        n_input_points=args.n_input_points,
    )

    m_off = load_snowflakenet(official_ckpt)
    m_our = load_snowflakenet(ours_ckpt)

    save_dir = args._resolved_save_dir.strip() or None  # noqa: SLF001

    for stem, ip, gp in slice_pairs:
        part = np.load(ip).astype(np.float32)
        gt = np.load(gp).astype(np.float32)

        pred_off = forward_from_npy(m_off, part, device, **fwd_kw)
        pred_our = forward_from_npy(m_our, part, device, **fwd_kw)

        cd_off = cdl1_times_1e3(pred_off, gt, device)
        cd_our = cdl1_times_1e3(pred_our, gt, device)

        _viz_four_columns(
            stem=stem,
            part=part,
            pred_official=pred_off,
            pred_ours=pred_our,
            gt=gt,
            cd_official_x1e3=cd_off,
            cd_ours_x1e3=cd_our,
            save_dir=save_dir,
            processed_root=processed_root,
            split=args.split,
            official_ckpt=official_ckpt,
            ours_ckpt=ours_ckpt,
        )

    del m_off, m_our
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
