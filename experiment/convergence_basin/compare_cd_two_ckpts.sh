#!/usr/bin/env bash
# 对同一预处理目录分别跑 03_eval_completion.py（预训练 vs 微调），便于对比 CD-L1×10³。
# 用法（在项目根）:
#   bash experiment/convergence_basin/compare_cd_two_ckpts.sh \
#     data/processed/PCN_far8_cano_in2048_gt16384 \
#     Snet/SnowflakeNet-main/completion/checkpoints/ckpt-best-pcn-cd_l1.pth \
#     checkpoints/snet_finetune/ckpt-best.pth
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${ROOT}/Snet/SnowflakeNet-main:${PYTHONPATH:-}"
PROC="${1:?arg1: processed-root}"
PRE="${2:?arg2: pretrained ckpt}"
FT="${3:?arg3: finetuned ckpt}"
echo "=== Pretrained ==="
python scripts/03_eval_completion.py --processed-root "$PROC" --split test --ckpt "$PRE"
echo "=== Finetuned ==="
python scripts/03_eval_completion.py --processed-root "$PROC" --split test --ckpt "$FT"
echo ""
echo "Done. 预期：微调平均 CD-L1×10³ 低于预训练；论文复现阶段常见相对降幅约 ≥20%（依 epoch 与数据而定，非硬阈值）。"
