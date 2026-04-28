"""
对单个 .pcd（PCN partial 或任意点云）做 SnowflakeNet 补全，使用 Snet 下预训练权重。

默认权重: Snet/SnowflakeNet-main/completion/checkpoints/ckpt-best-pcn-cd_l1.pth

**路径必须写全**，不要用 shell 里复制来的「...」省略段；本仓库下典型 partial 为：
  <项目根>/PCN/test/partial/<类别synset>/<模型id>/00.pcd
默认补全结束后会**弹出 Open3D 窗口**（红=输入，绿=补全）；仅保存、无显示时请加 --no-vis。

多视角最优 partial（与 ModelNet base_poses 不同，见 scripts/01_compute_best_view.py）:
  python3 scripts/01_compute_best_view.py --pcn-root PCN --splits test --out-json data/processed/PCN_hpr_best_views.json
  python3 scripts/04_infer_completion.py -i PCN/test/partial/02933112/<model_id>/00.pcd \\
    --pcn-best-view-json data/processed/PCN_hpr_best_views.json
  # -i 也可为目录 .../partial/<synset>/<model_id>，将选用 JSON 中的 best_partial_file。

用法示例:
  python3 scripts/04_infer_completion.py -i PCN/test/partial/02933112/1d7b35cda1bbd2e6eb1f243bab39fb29/00.pcd
（勿把说明文字里的「省略」写进命令；路径段名不能是三个点。）
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_SNET_ROOT = os.path.join(_PROJECT_ROOT, "Snet", "SnowflakeNet-main")
# 用于报错提示的合法示例（相对项目根，勿含省略号）
_EXAMPLE_PARTIAL_PCD = os.path.join(
    "PCN", "test", "partial", "02933112", "1d7b35cda1bbd2e6eb1f243bab39fb29", "00.pcd"
)
for _p in (_PROJECT_ROOT, _SNET_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# 与 simple_infer 一致；在导入 torch/open3d 前即可使用
_COMPLETED_DIR = os.path.join(_PROJECT_ROOT, "data", "completed")

_DEFAULT_CKPT = os.path.join(
    _SNET_ROOT,
    "completion",
    "checkpoints",
    "ckpt-best-pcn-cd_l1.pth",
)


def _fail_paths_help() -> str:
    abs_ex = os.path.join(_PROJECT_ROOT, _EXAMPLE_PARTIAL_PCD)
    return (
        f"  请使用真实存在的 .pcd 完整路径（勿含路径段 '...'）。\n"
        f"  仓库内示例: {abs_ex}\n"
        f"  或相对项目根: {_EXAMPLE_PARTIAL_PCD}"
    )


def _parse_pcn_partial_ctx(p: Path) -> tuple[str, str, str] | None:
    """从 .../<split>/partial/<synset>/<model_id>/... 解析三段；否则 None。"""
    parts = p.parts
    try:
        i = parts.index("partial")
    except ValueError:
        return None
    if i < 1 or i + 2 >= len(parts):
        return None
    return parts[i - 1], parts[i + 1], parts[i + 2]


def _load_pcn_best_views(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get("by_model") or {}


def _resolve_with_best_view_json(
    p: Path,
    by_model: dict[str, dict],
) -> Path | None:
    ctx = _parse_pcn_partial_ctx(p)
    if ctx is None:
        return None
    split, synset, model_id = ctx
    jk = f"{split}__{synset}__{model_id}"
    entry = by_model.get(jk)
    if not entry:
        return None
    name = entry.get("best_partial_file")
    if not name:
        return None
    partial_dir = p if p.is_dir() else p.parent
    cand = (partial_dir / name).resolve()
    return cand if cand.is_file() else None


def _resolve_and_check_pcd_input(
    raw: str,
    *,
    pcn_best_view_json: str | None = None,
) -> str:
    """展开路径；可为 .pcd 文件或 PCN partial 模型目录；可选按 JSON 换为最优视角 partial。"""
    s = (raw or "").strip()
    p = Path(os.path.abspath(os.path.expanduser(s)))
    parts = p.parts
    if any(part == "..." for part in parts) or "..." in s:
        print(
            "错误: 路径里不能包含省略号 '...'（那是文档里的占位，不是真实目录名）。\n"
            + _fail_paths_help(),
            file=sys.stderr,
        )
        sys.exit(2)
    if not p.exists():
        print(f"错误: 路径不存在: {p}\n" + _fail_paths_help(), file=sys.stderr)
        sys.exit(2)

    by_model: dict[str, dict] = {}
    if pcn_best_view_json:
        jp = Path(pcn_best_view_json).expanduser()
        if not jp.is_file():
            print(f"错误: --pcn-best-view-json 不是文件: {jp}", file=sys.stderr)
            sys.exit(2)
        by_model = _load_pcn_best_views(str(jp.resolve()))

    chosen: Path | None = None
    if p.is_dir():
        if by_model:
            chosen = _resolve_with_best_view_json(p, by_model)
        if chosen is None:
            cands = sorted(p.glob("*.pcd"))
            if not cands:
                print(f"错误: 目录下无 .pcd: {p}\n" + _fail_paths_help(), file=sys.stderr)
                sys.exit(2)
            chosen = cands[0]
            if by_model:
                print(f"[infer] 未在 JSON 中找到该模型键，使用目录内首文件: {chosen.name}")
    else:
        if not p.is_file():
            print(f"错误: 不是文件或目录: {p}\n" + _fail_paths_help(), file=sys.stderr)
            sys.exit(2)
        if not str(p).lower().endswith(".pcd"):
            print("错误: 输入需为 .pcd 文件或 partial 模型目录。", file=sys.stderr)
            sys.exit(2)
        chosen = p
        if by_model:
            swap = _resolve_with_best_view_json(p, by_model)
            if swap is not None and swap.resolve() != p.resolve():
                print(f"[infer] --pcn-best-view-json: 选用 {swap.name}（原输入 {p.name}）")
                chosen = swap

    assert chosen is not None
    return str(chosen)


def _load_pcd_xyz(path: str) -> np.ndarray:
    import open3d as o3d  # 延后导入，先做完路径检查

    pc = o3d.io.read_point_cloud(path)
    pts = np.asarray(pc.points, dtype=np.float64)
    if pts.size == 0:
        print(_fail_paths_help(), file=sys.stderr)
        raise ValueError(f"空点云或读点失败: {path}")
    return pts.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="单文件 .pcd → SnowflakeNet 补全（预训练/自定义 ckpt）。"
        " 默认在补全结束后打开 Open3D 可视化。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例（在项目根下执行）:\n  python3 scripts/04_infer_completion.py -i %s\n"
        % _EXAMPLE_PARTIAL_PCD,
    )
    ap.add_argument(
        "-i",
        "--input",
        required=True,
        help="输入 .pcd 的完整路径或相对 CWD 的路径（不要写 '...' 省略段）",
    )
    ap.add_argument(
        "-c",
        "--ckpt",
        default=_DEFAULT_CKPT,
        help="ckpt 路径（需含 'model' 的 state_dict）",
    )
    ap.add_argument(
        "-o",
        "--out-npy",
        default=None,
        help="补全结果 .npy；默认 data/completed/<输入主名>_completed.npy",
    )
    ap.add_argument(
        "--out-pcd",
        default=None,
        help="若指定，将补全点云再存为 .pcd",
    )
    ap.add_argument(
        "--no-vis",
        action="store_true",
        help="不弹窗（无显示器/SSH 无转发时用）；默认会弹出对比可视化",
    )
    ap.add_argument("--no-comp-filter", action="store_true")
    ap.add_argument("--sor-nb", type=int, default=20)
    ap.add_argument("--sor-std", type=float, default=2.0)
    ap.add_argument("--no-input-gate", action="store_true")
    ap.add_argument("--input-gate-mul", type=float, default=4.0)
    ap.add_argument("--input-knn", type=int, default=8)
    ap.add_argument("--filter-radius", type=float, default=0.0)
    ap.add_argument("--filter-radius-nb", type=int, default=16)
    ap.add_argument(
        "--export-stages-dir",
        default=None,
        help="导出各阶段 npy 的目录",
    )
    ap.add_argument(
        "--pcn-best-view-json",
        default=None,
        help="compute_pcn_best_hpr_view.py 输出的 JSON；-i 为 partial 下任一 .pcd 或该模型目录时，"
        "改用其中记录的 best_partial_file（多视角时）",
    )
    args = ap.parse_args()

    inp = _resolve_and_check_pcd_input(
        args.input, pcn_best_view_json=args.pcn_best_view_json
    )

    from src.pose_estimation.postprocess import FilterConfig  # noqa: E402
    from src.inference.complete_partial import complete_partial_points  # noqa: E402

    stem = os.path.splitext(os.path.basename(inp))[0]
    if args.out_npy:
        out_npy = args.out_npy
    else:
        os.makedirs(_COMPLETED_DIR, exist_ok=True)
        out_npy = os.path.join(_COMPLETED_DIR, f"{stem}_completed.npy")

    fcfg = FilterConfig(
        sor_nb=args.sor_nb,
        sor_std=args.sor_std,
        use_input_gate=not args.no_input_gate,
        input_knn=args.input_knn,
        input_gate_mul=args.input_gate_mul,
        radius=args.filter_radius,
        radius_nb=args.filter_radius_nb,
    )

    points = _load_pcd_xyz(inp)
    print(f"读取 {inp} 点数为 {points.shape[0]}")

    if not args.no_vis:
        print("补全结束后将打开 Open3D 窗口：左/红=输入，右/绿=补全（关闭窗口以结束）")

    complete_partial_points(
        points,
        args.ckpt,
        out_npy,
        show_vis=not args.no_vis,
        export_stages_dir=args.export_stages_dir,
        stage_stem=stem,
        do_comp_filter=not args.no_comp_filter,
        filter_cfg=fcfg,
    )

    if args.out_pcd:
        import open3d as o3d

        pred = np.load(out_npy)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pred.astype(np.float64))
        out_pcd = os.path.abspath(args.out_pcd)
        os.makedirs(os.path.dirname(out_pcd) or ".", exist_ok=True)
        o3d.io.write_point_cloud(out_pcd, pcd)
        print(f"已写 pcd: {out_pcd}")


if __name__ == "__main__":
    main()
