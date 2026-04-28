#!/usr/bin/env bash
# 冒烟：需已配置 PCN 根目录与 Snet。示例仅打印命令。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${ROOT}/Snet/SnowflakeNet-main:${PYTHONPATH:-}"
: "${PCN_ROOT:=${ROOT}/PCN}"
: "${OUT:=${ROOT}/data/processed/PCN_far_in2048_gt16384}"
echo "1) 预处理 T_far + bbox+PCA -> ${OUT}"
echo "   python ${ROOT}/scripts/00_preprocess_pcn.py --pcn-root ${PCN_ROOT} --out-root ${OUT}"
echo "2) 最佳视角 JSON"
echo "   python ${ROOT}/scripts/01_compute_best_view.py --pcn-root ${PCN_ROOT}"
echo "3) 训练 / 4) 评测 / 5) 位姿 见 README.md"
