"""
预处理 .npy 上对比：预训练 SnowflakeNet vs 微调 ckpt 的补全可视化（Open3D）。

**补全输入从哪读（核心参数 ``--input-source``）**

- **npy**：读预处理后的 ``{input_dir}/{stem}.npy``（bbox+PCA 后与训练/微调一致）。
- **pcd**：读 PCN **原始** partial（默认按 stem 拼 ``{PCN}/{split}/partial/{synset}/{model}/{view}.pcd``，
 见 ``--pcn-root``）；也可用 ``--pcd-file`` 任选路径；仍有 ``meta`` 可作最后备选。
- **both**：先弹窗 npy 会话，关闭后再弹 pcd 会话。

**.pcd 与 .npy 本身只是容器格式**；读入后都是 ``(N,3)`` float32 点。影响补全质量的是
**坐标系、是否 PCA 对齐、点数与重采样方式**，不是文件扩展名。  
npy 路径与 pcd 路径若表示**不同几何/不同系**（典型：预处理 canonical vs 原始物体系），
网络输出不可直接对比数值，只能分窗口看。

pcd 会话默认不叠画 ``gt/*.npy``（GT 为 PCA 后，与 pcd 推断系不一致）。

预处理影响分析（**仅预训练 ckpt**，单窗四云 + 可选 GT/CD；raw 与预处理默认 PCA 对齐显示，可用 ``--viz-no-align-raw`` 关）::

  PYTHONPATH=. python3 scripts/08_viz_completion_pretrain_vs_finetune.py \\
      --pretrain-ablation --pcn-root PCN

示例:
  PYTHONPATH=. python3 scripts/08_viz_completion_pretrain_vs_finetune.py
  PYTHONPATH=. python3 scripts/08_viz_completion_pretrain_vs_finetune.py --input-source both
  PYTHONPATH=. python3 scripts/08_viz_completion_pretrain_vs_finetune.py --input-source pcd \\
      --meta-dir data/processed/PCN_far_cano_in2048_gt16384/train/meta

无可显: ``--save-dir <目录>`` 写出 ply。

"""
from __future__ import annotations

import argparse
import os
import re
import sys
import zlib

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_SNET_ROOT = os.path.join(_PROJECT_ROOT, "Snet", "SnowflakeNet-main")
for _p in (_PROJECT_ROOT, _SNET_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.evaluation.npy_forward import forward_from_npy  # noqa: E402
from src.models.chamfer import chamfer_l1_symmetric  # noqa: E402
from src.models.snet_loader import load_snowflakenet  # noqa: E402
from src.utils.io import read_pcd_xyz, to_o3d_pcd  # noqa: E402


def _span(pts: np.ndarray) -> float:
    if pts.size == 0:
        return 1.0
    return float(np.ptp(pts, axis=0).max()) or 1.0


def _rigid_align_pca_to_reference(
    src: np.ndarray, ref: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """将 ``src`` 中与 ``ref`` 主轴不一致的旋转，用 PCA 正交对齐到 ``ref`` 坐标系。

    返回 ``(aligned_src_points, centroid_src, centroid_ref, R 3x3)``
    满足 ``aligned ≈ (src - cs) @ R.T + cr``（``R`` 为正交阵，det≈+1）。
    """
    sr = np.asarray(ref, dtype=np.float64)
    ss = np.asarray(src, dtype=np.float64)
    cs = ss.mean(axis=0)
    cr = sr.mean(axis=0)
    Xs = ss - cs
    Xr = sr - cr

    def axes_descending(X: np.ndarray) -> np.ndarray:
        cov = (X.T @ X) / max(len(X), 1)
        w, V = np.linalg.eigh(cov)
        order = np.argsort(w)[::-1]
        return V[:, order]

    Bs = axes_descending(Xs)
    Br = axes_descending(Xr)
    R = Br @ Bs.T
    if np.linalg.det(R) < 0:
        Bs2 = np.copy(Bs)
        Bs2[:, -1] *= -1.0
        R = Br @ Bs2.T
    aligned = (Xs @ R.T + cr).astype(np.float32)
    return aligned, cs.astype(np.float64), cr.astype(np.float64), R.astype(np.float64)


def _apply_same_rigid(pred: np.ndarray, cs: np.ndarray, cr: np.ndarray, R_np: np.ndarray) -> np.ndarray:
    """已知 ``align`` 对 partial 使用 ``(Xs)@R.T+cr``，对同系点云 pred 用相同 R, cs, cr。"""
    P = pred.astype(np.float64)
    R = np.asarray(R_np, dtype=np.float64)
    return ((P - cs) @ R.T + cr).astype(np.float32)


def _resample_2048(pts: np.ndarray, seed: int) -> np.ndarray:
    """与 ``complete_partial._resample_points`` 一致：原始 PCN partial → 2048 点送入网络。"""
    data = np.asarray(pts, dtype=np.float32)
    n_out = 2048
    rng = np.random.default_rng(seed & 0xFFFFFFFF)
    curr = data.shape[0]
    if curr < n_out:
        idx = rng.choice(curr, n_out, replace=True)
        return data[idx].astype(np.float32)
    if curr > n_out:
        idx = rng.choice(curr, n_out, replace=False)
        return data[idx].astype(np.float32)
    return data


def _default_pcn_partial_from_stem(stem: str, pcn_root: str) -> tuple[str | None, list[str]]:
    """
    按本仓库 sample_stem（见 src/data/naming.py）约定，从 stem 拼 PCN 官方 partial 路径：
    {pcn_root}/{split}/partial/{synset}/{model_id}/{view:02d}.pcd
    若不存在再试 {view}.pcd（无前导零）。
    """
    root = os.path.abspath(os.path.expanduser(pcn_root))
    parts = stem.split("__")
    if len(parts) < 5 or parts[0] != "pcn":
        return None, []
    split_n = parts[1]
    tax = parts[2]
    m = re.fullmatch(r"(.+)_(?P<syn>\d{8})$", tax)
    if not m:
        return None, []
    synset = m.group("syn")
    model_id = parts[3]
    vm = re.fullmatch(r"view_(?P<vi>\d+)", parts[4])
    if not vm:
        return None, []
    vi = int(vm.group("vi"))
    sub = os.path.join(root, split_n, "partial", synset, model_id)
    cands = [
        os.path.join(sub, f"{vi:02d}.pcd"),
        os.path.join(sub, f"{vi}.pcd"),
    ]
    tried: list[str] = []
    for c in cands:
        tried.append(c)
        if os.path.isfile(c):
            return c, tried
    return None, tried


def _meta_source_partial_path(meta_npz: str) -> str:
    z = np.load(meta_npz, allow_pickle=True)
    sp = z["source_partial"]
    if isinstance(sp, np.ndarray):
        sp = sp.item() if sp.size == 1 else sp[0]
    return str(sp)


def _resolve_meta_npz(args, stem: str, inp_dir: str, processed_root_abs: str) -> str | None:
    """候选路径依次尝试，命中第一个存在的 .npz。"""
    cands = []
    if getattr(args, "meta_dir", None):
        mdir = os.path.abspath(os.path.expanduser(args.meta_dir))
        cands.append(os.path.join(mdir, f"{stem}.npz"))
    if processed_root_abs:
        cands.append(os.path.join(processed_root_abs, args.split, "meta", f"{stem}.npz"))
    cands.extend(
        (
            os.path.join(_PROJECT_ROOT, "meta", f"{stem}.npz"),
            os.path.join(os.path.dirname(inp_dir), "meta", f"{stem}.npz"),
        )
    )
    for p in cands:
        if os.path.isfile(p):
            return p
    return None


def _run_pretrain_ablation_session(
    *,
    part_npy: np.ndarray,
    part_pcd: np.ndarray,
    pred_from_npy: np.ndarray,
    pred_from_pcd: np.ndarray,
    gt_arr: np.ndarray | None,
    stem: str,
    inp_dir: str,
    args,
    device,
) -> None:
    """
    仅预训练 ckpt：同一物体「预处理输入 vs 原始 PCN partial」两次补全，单窗并排；
    GT 仅在可与 pred_from_npy 对齐时绘制并打印 Chamfer(pred_npy, GT)。
    默认将 raw partial 及其补全做 PCA 刚性对齐到预处理 partial 的坐标系（仅影响显示与布局）。
    """
    part_pcd_vis = part_pcd
    pred_from_pcd_vis = pred_from_pcd
    if not getattr(args, "viz_no_align_raw", False):
        part_pcd_vis, cs, cr, R = _rigid_align_pca_to_reference(part_pcd, part_npy)
        pred_from_pcd_vis = _apply_same_rigid(pred_from_pcd, cs, cr, R)
        viz_align_line = (
            "  可视化: raw 分支已 PCA 对齐到预处理朝向（仅显示；Chamfer 仍为 preproc 路径）"
        )
    else:
        viz_align_line = "  可视化: raw 分支未对齐（--viz-no-align-raw）"

    dx = max(
        _span(part_npy),
        _span(part_pcd_vis),
        _span(pred_from_npy),
        _span(pred_from_pcd_vis),
        _span(gt_arr) if gt_arr is not None else 1.0,
    )
    sep = dx * 1.55
    blobs = [
        (part_npy, (0.95, 0.15, 0.15)),
        (pred_from_npy, (1.0, 0.5, 0.05)),
        (part_pcd_vis, (0.75, 0.2, 0.85)),
        (pred_from_pcd_vis, (0.15, 0.85, 0.35)),
    ]
    labels = ["input_preproc", "pretrain_from_preproc", "input_raw_pcd", "pretrain_from_raw"]
    if gt_arr is not None:
        blobs.append((gt_arr, (0.35, 0.45, 0.95)))
        labels.append("gt_preproc_system")
    n_blob = len(blobs)
    x0 = -(n_blob - 1) * 0.5 * sep
    geoms = []
    for i, lab in enumerate(labels):
        pts, col = blobs[i]
        gx = float(x0 + i * sep)
        geoms.append((lab, np.asarray(pts + np.array([gx, 0, 0], dtype=np.float32)), col))

    print("")
    print("=== [--pretrain-ablation] 仅预训练 ckpt：预处理输入 vs PCN 原始 partial ===")
    print(f"stem: {stem}")
    print(f"  ckpt (pretrain only): {args.ckpt_pretrain}")
    print(
        "  左→右: 红=预处理 partial, 橙=pretrain 补全 | 紫=raw partial, 绿=pretrain 补全"
        + (", 蓝=GT(与预处理系一致)" if gt_arr is not None else ""),
    )
    print(viz_align_line)

    if gt_arr is not None:
        pn = torch.from_numpy(pred_from_npy).float().unsqueeze(0).to(device)
        g = torch.from_numpy(gt_arr).float().unsqueeze(0).to(device)
        cd = chamfer_l1_symmetric(pn, g).item()
        print(f"  Chamfer L1 (pretrain@preproc vs GT，同坐标系): {cd:.6f}")
        print("  （raw 分支与 GT 不同坐标系，不计算 CD）")
    else:
        print("  无 gt：未打印数值对比")

    if args.save_dir:
        out_root = os.path.abspath(args.save_dir)
        os.makedirs(out_root, exist_ok=True)
        import open3d as o3d

        base = "".join((c if c.isalnum() or c in "-_" else "_" for c in stem))[:180]
        for lab, pts, _c in geoms:
            path = os.path.join(out_root, f"{base}__pretrain_ablation__{lab}.ply")
            o3d.io.write_point_cloud(path, to_o3d_pcd(np.asarray(pts)))
            print(f"  wrote {path}")
        return

    import open3d as o3d

    o3d_geoms = [to_o3d_pcd(pts, color=c) for _name, pts, c in geoms]
    o3d.visualization.draw_geometries(
        o3d_geoms,
        window_name=f"pretrain_ablation | {stem[:50]}",
        width=1480,
        height=760,
    )


def _run_one_visual_session(
    *,
    part: np.ndarray,
    gt_arr: np.ndarray | None,
    session_id: str,
    session_note: str,
    stem: str,
    inp_dir: str,
    args,
    m_pre,
    m_ft,
    device,
    kwargs: dict,
) -> None:
    pred_pre = forward_from_npy(m_pre, part, device, **kwargs)
    pred_ft = forward_from_npy(m_ft, part, device, **kwargs)

    dx = max(
        _span(part),
        _span(pred_pre),
        _span(pred_ft),
        _span(gt_arr) if gt_arr is not None else 1.0,
    )
    sep = dx * 1.6
    blobs = [
        (part, (0.95, 0.2, 0.15)),
        (pred_pre, (1.0, 0.55, 0.1)),
        (pred_ft, (0.25, 0.85, 0.35)),
    ]
    labels = ["input", "pretrain_pred", "finetune_pred"]
    if gt_arr is not None:
        blobs.append((gt_arr, (0.35, 0.45, 0.95)))
        labels.append("gt")
    n_blob = len(blobs)
    x0 = -(n_blob - 1) * 0.5 * sep

    geoms = []
    for i, label in enumerate(labels):
        pts, col = blobs[i]
        gx = float(x0 + i * sep)
        geoms.append((label, np.asarray(pts + np.array([gx, 0, 0], dtype=np.float32)), col))

    o3d_geoms = [to_o3d_pcd(pts, color=c) for _name, pts, c in geoms]

    print("")
    print(f"--- [{session_id}] {session_note} ---")
    print(f"stem: {stem}")
    if args.processed_root:
        print(f"  layout: processed-root -> {inp_dir}")
    else:
        print(f"  layout: flat input-dir -> {inp_dir}")
    print(f"  pretrain: {args.ckpt_pretrain}")
    print(f"  finetune: {args.ckpt_finetune}")
    print(
        "  左→右: 红=input, 橙=pretrain, 绿=finetune"
        + (", 蓝=GT" if gt_arr is not None else ""),
    )

    if args.save_dir:
        out_root = os.path.abspath(args.save_dir)
        os.makedirs(out_root, exist_ok=True)
        import open3d as o3d

        base = "".join((c if c.isalnum() or c in "-_" else "_" for c in stem))[:180]
        for label, pts, _c in geoms:
            path = os.path.join(out_root, f"{base}__{session_id}__{label}.ply")
            o3d.io.write_point_cloud(path, to_o3d_pcd(np.asarray(pts)))
            print(f"  wrote {path}")
        return

    import open3d as o3d

    o3d.visualization.draw_geometries(
        o3d_geoms,
        window_name=f"[{session_id}] pretrain_vs_finetune | {stem[:52]}",
        width=1400,
        height=780,
    )


def _pick_stem(inp_dir: str, stem: str | None, index: int) -> str:
    if stem:
        return stem
    names = sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(inp_dir)
        if f.endswith(".npy")
    )
    if not names:
        raise FileNotFoundError(f"无 .npy: {inp_dir}")
    idx = index % len(names)
    return names[idx]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="预训练 vs 微调：同一 input .npy 补全并排可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--input-dir",
        default=os.path.join(_PROJECT_ROOT, "input"),
        help="直接包含待补全 .npy 的目录（默认：<项目根>/input）",
    )
    ap.add_argument(
        "--gt-dir",
        default=os.path.join(_PROJECT_ROOT, "gt"),
        help="与 stem 同名的 GT .npy 目录（默认：<项目根>/gt；文件不存在则跳过）",
    )
    ap.add_argument(
        "--processed-root",
        default="",
        help="若非空：改为从 {root}/{split}/input 读入，并忽略 --input-dir / --gt-dir",
    )
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--stem", default=None, help="不含 .npy；缺省则用 --sample-index")
    ap.add_argument("--sample-index", type=int, default=0, help="按名字排序后的第几条")
    ap.add_argument(
        "--ckpt-pretrain",
        "--official-ckpt",
        default=os.path.join(
            _SNET_ROOT, "completion", "checkpoints", "ckpt-best-pcn-cd_l1.pth"
        ),
        help="SnowflakeNet 官方 PCN 预训练权重",
    )
    ap.add_argument(
        "--ckpt-finetune",
        "--ours-ckpt",
        default=os.path.join(_PROJECT_ROOT, "checkpoints", "ckpt-best.pth"),
        help="微调 best 权重（默认项目根 checkpoints/ckpt-best.pth）",
    )
    ap.add_argument("--input-mode", choices=("direct", "upsample", "legacy"), default="direct")
    ap.add_argument("--n-input-points", type=int, default=2048)
    ap.add_argument("--comp-filter", action="store_true", help="与 eval 一致的补全后 SOR/gate（默认关）")
    ap.add_argument(
        "--no-gt",
        action="store_true",
        help="不加载 gt（若不存在也会自动跳过）",
    )
    ap.add_argument(
        "--save-dir",
        default="",
        help="若非空：写出 ply（input/pretrain_pred/finetune_pred/gt）后退出（不弹窗）",
    )
    ap.add_argument(
        "--input-source",
        default="npy",
        choices=("npy", "pcd", "both"),
        help="补全网络输入从哪里读：**npy**=预处理 `{stem}.npy`；**pcd**=原始 partial .pcd（--pcd-file 或 meta）；"
        "**both**=先做 npy 会话再做 pcd 会话",
    )
    ap.add_argument(
        "--completion-input",
        default=None,
        choices=("processed", "raw_pcn", "both"),
        metavar="DEPRECATED",
        help="弃用：processed=npy，raw_pcn=pcd，both=both（请改用 --input-source）",
    )
    ap.add_argument(
        "--pcd-file",
        default="",
        help="原始 partial .pcd；若设置则最优先（仅 input-source 含 pcd）",
    )
    ap.add_argument(
        "--pcn-root",
        default=os.path.join(_PROJECT_ROOT, "PCN"),
        help="官方 PCN 解压根目录；在未指定 --pcd-file 时，按 stem 拼接 partial 路径（默认项目下 PCN/）",
    )
    ap.add_argument(
        "--meta-dir",
        default="",
        help="存放 {stem}.npz（含字段 source_partial 指向原始 .pcd）；"
        "未设时依次尝试 processed.../meta、项目 meta/、与 input 同级 meta/",
    )
    ap.add_argument(
        "--raw-resample-seed",
        type=int,
        default=-1,
        help=".pcd 分支：随机重采样到 2048 的随机种子；-1=用 stem 的 adler32",
    )
    ap.add_argument(
        "--pretrain-ablation",
        action="store_true",
        help="仅加载预训练 ckpt，同一 stem 单窗并排：预处理 input.npy 与原始 partial 的两次补全（看预处理对预训练输出的影响）；隐含 --input-source both",
    )
    ap.add_argument(
        "--viz-no-align-raw",
        action="store_true",
        help="(--pretrain-ablation) 不对 raw partial/其补全做 PCA 对齐到预处理系；默认对齐以便五列朝向一致",
    )
    args = ap.parse_args()
    # 兼容旧参数 --completion-input（先于 ablation）
    if getattr(args, "completion_input", None) is not None:
        legacy_map = {"processed": "npy", "raw_pcn": "pcd", "both": "both"}
        args.input_source = legacy_map[args.completion_input]
    if getattr(args, "pretrain_ablation", False):
        args.input_source = "both"

    if args.processed_root:
        root = os.path.abspath(os.path.expanduser(args.processed_root))
        inp_dir = os.path.join(root, args.split, "input")
        gt_dir = os.path.join(root, args.split, "gt")
    else:
        inp_dir = os.path.abspath(os.path.expanduser(args.input_dir))
        gt_dir = os.path.abspath(os.path.expanduser(args.gt_dir))

    if not os.path.isdir(inp_dir):
        raise FileNotFoundError(f"input 目录不存在: {inp_dir}")

    stem = _pick_stem(inp_dir, args.stem, args.sample_index)
    inp_path = os.path.join(inp_dir, f"{stem}.npy")

    _ckpts_check = [args.ckpt_pretrain]
    if not getattr(args, "pretrain_ablation", False):
        _ckpts_check.append(args.ckpt_finetune)
    for p in _ckpts_check:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"权重不存在: {p}")

    need_npy = args.input_source in ("npy", "both")
    part_canonical = None
    if need_npy:
        if not os.path.isfile(inp_path):
            raise FileNotFoundError(
                f"需要预处理 input .npy: {inp_path}\n"
                "（若只需原始 PCN 补全，请 ``--input-source pcd``，并保证 ``--pcn-root`` 下有对应 partial）",
            )
        part_canonical = np.load(inp_path).astype(np.float32)
    gt_arr: np.ndarray | None = None
    if not args.no_gt:
        gpath = os.path.join(gt_dir, f"{stem}.npy")
        if os.path.isfile(gpath):
            gt_arr = np.load(gpath).astype(np.float32)

    pr_abs = os.path.abspath(os.path.expanduser(args.processed_root)) if args.processed_root else ""
    meta_npz = _resolve_meta_npz(args, stem, inp_dir, pr_abs)
    seed_raw = (
        args.raw_resample_seed
        if args.raw_resample_seed >= 0
        else int(zlib.adler32(stem.encode("utf-8")) % (2**32))
    )

    sessions = []
    if need_npy:
        sessions.append((
            "npy",
            part_canonical,
            "预处理 input.npy（bbox+PCA 对齐，与训练/微调一致）",
        ))

    need_pcd = args.input_source in ("pcd", "both")
    if need_pcd:
        src_pcd = ""
        note_head = ""
        if getattr(args, "pcd_file", None) and str(args.pcd_file).strip():
            src_pcd = os.path.abspath(os.path.expanduser(args.pcd_file))
            if not os.path.isfile(src_pcd):
                raise FileNotFoundError(f"--pcd-file 不存在: {src_pcd}")
            note_head = (
                "原始 PCN partial（--pcd-file）→ 重采样 2048\n"
                f"  pcd={src_pcd}\n"
                f"  resample_seed={seed_raw}"
            )
        elif args.pcn_root:
            found, tried_pcn = _default_pcn_partial_from_stem(stem, args.pcn_root)
            if found:
                src_pcd = found
                note_head = (
                    "原始 PCN partial（按 stem + --pcn-root 自动拼路径）→ 重采样 2048\n"
                    f"  pcn_root={os.path.abspath(os.path.expanduser(args.pcn_root))}\n"
                    f"  pcd={src_pcd}\n"
                    f"  resample_seed={seed_raw}"
                )
            elif meta_npz:
                src_pcd = _meta_source_partial_path(meta_npz)
                if not os.path.isfile(src_pcd):
                    raise FileNotFoundError(f"meta 中 source_partial 路径不存在: {src_pcd}")
                note_head = (
                    "原始 PCN partial（meta source_partial）→ 重采样 2048\n"
                    f"  meta={meta_npz}\n"
                    f"  source_partial={src_pcd}\n"
                    f"  resample_seed={seed_raw}"
                )
            else:
                tried_meta = []
                if args.meta_dir:
                    tried_meta.append(
                        os.path.join(os.path.abspath(os.path.expanduser(args.meta_dir)), f"{stem}.npz"),
                    )
                if pr_abs:
                    tried_meta.append(os.path.join(pr_abs, args.split, "meta", f"{stem}.npz"))
                tried_meta.extend([
                    os.path.join(_PROJECT_ROOT, "meta", f"{stem}.npz"),
                    os.path.join(os.path.dirname(inp_dir), "meta", f"{stem}.npz"),
                ])
                uniq_meta = list(dict.fromkeys(tried_meta))
                msg = (
                    f"找不到该样本对应的原始 partial：\n"
                    f"  已按 stem 尝试 PCN 路径（--pcn-root={args.pcn_root}）：\n"
                )
                for t in tried_pcn:
                    msg += f"    {t}\n"
                msg += (
                    "  请确认已下载并解压官方 PCN 到 ``--pcn-root``，或使用 ``--pcd-file`` 指定 .pcd。\n"
                    f"  （可选）若有预处理 meta，已尝试：\n"
                )
                for t in uniq_meta:
                    msg += f"    {t}\n"
                raise FileNotFoundError(msg)
        else:
            raise RuntimeError("internal: need_pcd but pcn_root empty")
        raw_xyz = read_pcd_xyz(src_pcd)
        part_raw = _resample_2048(raw_xyz, seed_raw)
        sessions.append(("pcd", part_raw, note_head))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = dict(
        input_mode=args.input_mode,
        do_comp_filter=args.comp_filter,
        n_input_points=args.n_input_points,
    )

    print(f"stem: {stem}  sample_index={args.sample_index}  input-source={args.input_source}")

    if getattr(args, "pretrain_ablation", False):
        if (
            len(sessions) != 2
            or sessions[0][0] != "npy"
            or sessions[1][0] != "pcd"
        ):
            raise RuntimeError(
                "--pretrain-ablation 要求同时加载预处理 .npy 与 PCN 原始 partial；"
                "请检查 input目录、--pcn-root/--pcd-file。",
            )
        part_npy = sessions[0][1]
        part_rc = sessions[1][1]
        m_only = load_snowflakenet(os.path.abspath(args.ckpt_pretrain))
        pred_np = forward_from_npy(m_only, part_npy, device, **kwargs)
        pred_rc = forward_from_npy(m_only, part_rc, device, **kwargs)
        _run_pretrain_ablation_session(
            part_npy=part_npy,
            part_pcd=part_rc,
            pred_from_npy=pred_np,
            pred_from_pcd=pred_rc,
            gt_arr=gt_arr,
            stem=stem,
            inp_dir=inp_dir,
            args=args,
            device=device,
        )
        return 0

    m_pre = load_snowflakenet(os.path.abspath(args.ckpt_pretrain))
    m_ft = load_snowflakenet(os.path.abspath(args.ckpt_finetune))

    for session_id, part, session_note in sessions:
        # pcd：物体系 partial；gt.npy 为 PCA 后，不能与 pcd 推断同屏对齐
        gt_use = gt_arr if session_id != "pcd" else None
        note_out = session_note
        if session_id == "pcd" and gt_arr is not None:
            note_out = session_note + (
                "\n  （本窗口不显示 GT：gt.npy 为 PCA 后坐标，与 raw 推断系不一致）"
            )
        _run_one_visual_session(
            part=part,
            gt_arr=gt_use,
            session_id=session_id,
            session_note=note_out,
            stem=stem,
            inp_dir=inp_dir,
            args=args,
            m_pre=m_pre,
            m_ft=m_ft,
            device=device,
            kwargs=kwargs,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
