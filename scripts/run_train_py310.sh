#!/usr/bin/env bash
# 在 Python>=3.10 环境中调用微调脚本（与 spd/Python3.7 区分）。
# 用法（先 conda activate snowflake-wandb 再执行）:
#   bash scripts/run_train_py310.sh [--no-wandb] ...
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/Snet/SnowflakeNet-main:${PYTHONPATH:-}"

python - << 'PY' || exit 1
import sys
if sys.version_info < (3, 8):
    print(
        "错误：当前 Python %s。wandb_v1 密钥需要 wandb>=0.22.3，仅支持 Python>=3.8。\n"
        "请: conda activate <你的 py310 环境> 或 conda env create -f environment_wandb_online.yml\n"
        "详见 README「AutoDL / 在线 W&B」。"
        % ".".join(map(str, sys.version_info[:3])),
        file=sys.stderr,
    )
    sys.exit(2)
PY

exec python scripts/02_train_completion.py "$@"
