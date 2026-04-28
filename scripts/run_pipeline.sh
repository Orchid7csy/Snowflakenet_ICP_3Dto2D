#!/usr/bin/env bash
# 冒烟：需已配置 PCN 根目录与 Snet。示例仅打印命令。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${ROOT}/Snet/SnowflakeNet-main:${PYTHONPATH:-}"
: "${PCN_ROOT:=${ROOT}/PCN}"
: "${OUT:=${ROOT}/data/processed/PCN_far8_cano_in2048_gt16384}"
: "${OUT_EASY:=${ROOT}/data/processed/PCN_hpr_easy_cano_in2048_gt16384}"
echo "1) 预处理：固定 8 视角 + T_far + normalize_by_complete -> ${OUT}"
echo "   python ${ROOT}/scripts/00_preprocess_pcn.py --pcn-root ${PCN_ROOT} --out-root ${OUT}"
echo "2) （可选）最佳视角 JSON，用于 curriculum easy 子集"
echo "   python ${ROOT}/scripts/01_compute_best_view.py --pcn-root ${PCN_ROOT} --splits train,val,test"
echo "3) curriculum easy：仍可用旧流程单独产出 ${OUT_EASY}（见 README），或与 hard 共用同一 processed 根关闭课程学习"
echo "   # 例如仅 hard：python ${ROOT}/scripts/02_train_completion.py --data-root-hard ${OUT}"
echo "4) 评测 / 位姿估计见 README.md"
