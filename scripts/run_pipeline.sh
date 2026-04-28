#!/usr/bin/env bash
# 冒烟：需已配置 PCN 根目录与 Snet。示例仅打印命令。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${ROOT}:${ROOT}/Snet/SnowflakeNet-main:${PYTHONPATH:-}"
: "${PCN_ROOT:=${ROOT}/PCN}"
: "${OUT:=${ROOT}/data/processed/PCN_far_cano_in2048_gt16384}"
: "${OUT_EASY:=${ROOT}/data/processed/PCN_hpr_easy_cano_in2048_gt16384}"
echo "1) 预处理 T_far + bbox+PCA -> ${OUT}"
echo "   python ${ROOT}/scripts/00_preprocess_pcn.py --pcn-root ${PCN_ROOT} --out-root ${OUT}"
echo "2) 最佳视角 JSON"
echo "   python ${ROOT}/scripts/01_compute_best_view.py --pcn-root ${PCN_ROOT} --splits train,val,test"
echo "3) 预处理 curriculum easy 集 -> ${OUT_EASY}"
echo "   python ${ROOT}/scripts/00_preprocess_pcn.py --pcn-root ${PCN_ROOT} --mode easy --out-root ${OUT_EASY}"
echo "4) curriculum 微调"
echo "   python ${ROOT}/scripts/02_train_completion.py --data-root-easy ${OUT_EASY} --data-root-hard ${OUT}"
echo "5) 评测 / 6) 位姿 见 README.md"
