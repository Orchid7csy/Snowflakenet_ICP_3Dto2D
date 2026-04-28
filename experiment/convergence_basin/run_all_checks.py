#!/usr/bin/env python3
"""
一键运行 ``experiment/convergence_basin/checks/`` 下的核查脚本（不含 pytest）。

用法::
  cd <项目根>
  export PYTHONPATH="$(pwd):$(pwd)/Snet/SnowflakeNet-main:${PYTHONPATH:-}"
  python experiment/convergence_basin/run_all_checks.py [--processed-root DIR] [--skip-processed]

若提供 ``--processed-root``，会运行 ``verify_processed_dataset.py``（否则跳过）。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
_CHECKS = os.path.join(_HERE, "checks")


def _run(py: str, args: list[str]) -> int:
    cmd = [sys.executable, os.path.join(_CHECKS, py)] + args
    print("+", " ".join(cmd))
    return subprocess.call(cmd, cwd=_ROOT, env={**os.environ, "PYTHONPATH": _py_path()})


def _py_path() -> str:
    snet = os.path.join(_ROOT, "Snet", "SnowflakeNet-main")
    cur = os.environ.get("PYTHONPATH", "")
    parts = [_ROOT, snet]
    if cur:
        parts.append(cur)
    return os.pathsep.join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="运行 convergence_basin 内置核查")
    ap.add_argument(
        "--processed-root",
        default="",
        help="若指定且存在，则核查预处理目录（否则跳过 verify_processed_dataset）",
    )
    ap.add_argument("--split", default="test")
    ap.add_argument("--skip-processed", action="store_true")
    ap.add_argument("--training-data-root", default="", help="若指定则跑 verify_training_canonical_space")
    args = ap.parse_args()

    os.environ["PYTHONPATH"] = _py_path()

    rc = 0
    rc |= _run("verify_decanonicalize_row_major.py", [])

    pr = args.processed_root.strip()
    if not args.skip_processed and pr:
        pr_abs = pr if os.path.isabs(pr) else os.path.join(_ROOT, pr)
        if os.path.isdir(pr_abs):
            rc |= _run(
                "verify_processed_dataset.py",
                ["--processed-root", pr_abs, "--split", args.split],
            )
        else:
            print(f"[skip] processed-root 不存在: {pr_abs}", file=sys.stderr)

    tr = args.training_data_root.strip()
    if tr:
        tr_abs = tr if os.path.isabs(tr) else os.path.join(_ROOT, tr)
        rc |= _run(
            "verify_training_canonical_space.py",
            ["--data-root", tr_abs, "--split", "train", "--dry-run"],
        )

    return min(rc, 255)


if __name__ == "__main__":
    raise SystemExit(main())
