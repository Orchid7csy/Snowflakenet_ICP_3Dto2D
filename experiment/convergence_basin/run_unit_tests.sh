#!/usr/bin/env bash
# 运行仓库单元测试（避免 ROS pytest 插件干扰时可先 export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/Snet/SnowflakeNet-main:${PYTHONPATH:-}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD="${PYTEST_DISABLE_PLUGIN_AUTOLOAD:-1}"
exec python3 -m pytest tests/ "$@"
