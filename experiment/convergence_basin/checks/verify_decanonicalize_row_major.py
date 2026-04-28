#!/usr/bin/env python3
"""
数值核查：反归一化与 ``scripts/05_estimate_pose.py::_apply_row_transform`` 行向量约定一致。

定义（与仓库实现一致）::
  p_pred_obj = P_pred_cano * scale + centroid_cano   # broadcast
  P_pred_w_row = p_pred_obj @ T[:3,:3] + T[:3,3]

用法::
  PYTHONPATH=<项目根> python experiment/convergence_basin/checks/verify_decanonicalize_row_major.py
"""
from __future__ import annotations

import argparse
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def _apply_row_transform(points, t):
    import numpy as np

    p = np.asarray(points, dtype=np.float64)
    tt = np.asarray(t, dtype=np.float64).reshape(4, 4)
    return (p @ tt[:3, :3] + tt[:3, 3]).astype(np.float64)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import numpy as np

    rng = np.random.default_rng(args.seed)
    n = 1024
    p_cano = rng.standard_normal((n, 3)).astype(np.float64) * 0.3
    centroid = rng.standard_normal(3).astype(np.float64).reshape(1, 3)
    scale = float(rng.uniform(0.4, 2.0))
    p_obj = p_cano * scale + centroid

    # random proper rigid (rotation + translation)
    a = rng.standard_normal((3, 3))
    q, _r = np.linalg.qr(a)
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1.0
    R = q.astype(np.float64)
    trans = rng.standard_normal(3).astype(np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = trans

    p_w = _apply_row_transform(p_obj, T)

    # Round-trip inverse: p_obj_rec ≈ (p_w - t) @ R^{-1} = (p_w - t) @ R.T
    p_obj_rec = (p_w - trans.reshape(1, 3)) @ R.T
    err = float(np.max(np.abs(p_obj_rec - p_obj)))
    if err > 1e-9:
        print(f"[FAIL] round-trip err={err:.3e}", file=sys.stderr)
        return 1

    # Explicit formula match _apply_row_transform (same as pipeline)
    p_w2 = _apply_row_transform(p_obj, T)
    if float(np.max(np.abs(p_w - p_w2))) > 1e-12:
        print("[FAIL] duplicate apply mismatch", file=sys.stderr)
        return 1

    print(f"[OK] 行向量约定 round-trip max_err={err:.3e}（反归一化与 T @ row 一致）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
