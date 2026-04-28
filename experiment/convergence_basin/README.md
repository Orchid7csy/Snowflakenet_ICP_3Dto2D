# 收敛域扩张（Convergence Basin）实验包

本目录汇总 **预处理核查、反归一化数值自检、训练空间说明、一键核查入口**。  
**完整实验顺序、穿插验证点与预期指标**见项目根目录 **`README.md`**。

## 目录结构

| 路径 | 说明 |
|------|------|
| `checks/verify_processed_dataset.py` | 核查 `input/gt/obs_w/meta` 形状与 meta 字段 |
| `checks/verify_decanonicalize_row_major.py` | 行向量约定与齐次变换自检 |
| `checks/verify_training_canonical_space.py` | 确认损失仅在 canonical 空间（数据集路径说明） |
| `run_all_checks.py` | 依次调用上述脚本（可选 `--processed-root`） |
| `run_unit_tests.sh` | 调用仓库 `tests/`（需 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 时见脚本注释） |
| `compare_cd_two_ckpts.sh` | 对同一 `processed-root` 分别评测预训练与微调 checkpoint 的 CD |

## 最短用法

在项目根执行：

```bash
export PYTHONPATH="$(pwd):$(pwd)/Snet/SnowflakeNet-main:${PYTHONPATH:-}"

# 数值约定（无数据亦可）
python experiment/convergence_basin/checks/verify_decanonicalize_row_major.py

# 若有预处理产物
python experiment/convergence_basin/checks/verify_processed_dataset.py \
  --processed-root data/processed/PCN_far8_cano_in2048_gt16384 --split test --max-stems 50

# 聚合
python experiment/convergence_basin/run_all_checks.py \
  --processed-root data/processed/PCN_far8_cano_in2048_gt16384
```

完整流程与验证清单见 **`README.md`（项目根目录）**。
