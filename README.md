# SnowflakeNet + FPFH/ICP — PCN 收敛域扩张完整实验

本仓库实现 **基于生成式几何先验的位姿收敛域扩张**：在极端残缺条件下，将 SnowflakeNet（SNet）点云补全作为 **几何代理（Geometric Proxy）**，在 ICP 原生吸引域外构造更接近完整的观测，引导优化轨迹进入收敛半径；**最终位姿精度仍由原始观测与 CAD 的 ICP 锁定**，补全网络不作为毫米级精度来源。

**一站式核查脚本与自检工具**位于 **`experiment/convergence_basin/`**（详见该目录下 `README.md`）。  
**Composer / Thinking / Skipped 卡顿**：可复制提示词见 **`experiment/convergence_basin/CURSOR_THINKING_SKIP.md`**，规则摘要见 **`.cursor/rules/avoid-thinking-stall.mdc`**（`.cursorignore` 已排除缓存与 `data/processed/` 等以降低索引量）。

---

## 目录与依赖

| 路径 | 说明 |
|------|------|
| `scripts/00_preprocess_pcn.py` | PCN `.pcd` → canonical `input/`、`gt/`、`obs_w/`、`meta/`（固定 8 视角，`normalize_by_complete`） |
| `scripts/01_compute_best_view.py` | （可选）每模型 HPR 最优视角 JSON，用于 curriculum **easy** 子集 |
| `scripts/02_train_completion.py` | SnowflakeNet 微调（Encoder/Decoder 分组学习率；默认加载预训练 `ckpt-best-pcn-cd_l1.pth`） |
| `scripts/03_eval_completion.py` | 预处理 `.npy` 上 CD-L1×10³ |
| `scripts/04_infer_completion.py` | 单 `.pcd` 补全 |
| `scripts/05_estimate_pose.py` | FPFH 粗配 → canonical 补全 → **反归一化** → Gate-ICP → 最终 ICP（`obs_w` vs CAD） |
| `scripts/07_viz_convergence_basin.py` | 双窗格可视化（Canonical 补全质量 / World 位姿对齐），等价于带 `--vis` 的位姿管线 |
| `scripts/08_viz_completion_pretrain_vs_finetune.py` | 预训练 vs 微调补全对比 |
| `scripts/09_verify_cano_vs_legacy.py` | canonical vs 旧 PCA 管线坍缩对比 |
| `Snet/SnowflakeNet-main/` | 需本地克隆 SnowflakeNet（见 `.gitignore`） |
| `experiment/convergence_basin/` | **预处理形状/meta 核查、行向量反归一化自检、训练空间说明、`run_all_checks.py`** |

公共代码在 `src/`。姊妹仓 legacy：`../SnowflakeNet_FPFH_ICP_legacy`。

---

## 完整实验流程（顺序与穿插验证）

按下表顺序执行。**验证列**为建议在完成该步骤后立即运行的检查；**预期效果**为判断实验是否「在轨道上」的经验阈值（非替代论文原始指标）。

### 阶段 0 — 环境与代码就绪

| 动作 | 命令 / 说明 |
|------|-------------|
| 克隆 SnowflakeNet | 置于 `Snet/SnowflakeNet-main/`，含 `completion/checkpoints/ckpt-best-pcn-cd_l1.pth` |
| PCN 数据 | 解压到项目内 `PCN/`（或自建 `--pcn-root`） |

**验证：**无。

**预期：**`python -c "import torch; print(torch.__version__)"` 可运行；GPU 可选。

---

### 阶段 1 — 预处理（canonical + `T_far` + 固定点数）

| 动作 | 命令示例 |
|------|----------|
| 生成主集合（固定 8 视角 `view_idx∈[0,7]`） | `python scripts/00_preprocess_pcn.py --pcn-root PCN --splits train,val,test --workers 8` |

默认输出目录：`data/processed/PCN_far8_cano_in2048_gt16384/`（可用 `--out-root` 覆盖）。

**穿插验证 A — 数据契约（强烈推荐）**

```bash
export PYTHONPATH="$(pwd):$(pwd)/Snet/SnowflakeNet-main:${PYTHONPATH:-}"

python experiment/convergence_basin/checks/verify_processed_dataset.py \
  --processed-root data/processed/PCN_far8_cano_in2048_gt16384 --split test
```

也可：`python experiment/convergence_basin/run_all_checks.py --processed-root data/processed/PCN_far8_cano_in2048_gt16384`

| 检查项 | 预期效果（Go） |
|--------|----------------|
| `input/*.npy` 形状 | 严格 `(2048, 3)` |
| `gt/*.npy` 形状 | 严格 `(16384, 3)`（与 `--num-gt 16384` 一致时） |
| `obs_w/*.npy` 形状 | `(2048, 3)` |
| `meta/*.npz` | 含 `centroid_cano` 或 `C_cano`、`scale_cano`、`view_idx`，且 `view_idx∈[0,7]` |

---

### 阶段 2 —（可选）课程学习 easy 集与最佳视角

若使用 easy→hard curriculum：先跑 `01_compute_best_view.py` 得到 JSON，再按旧文档单独生成 **HPR easy** 预处理目录（例如 `PCN_hpr_easy_cano_in2048_gt16384`）。  
主实验亦可 **不使用 curriculum**：令 `--data-root-easy` 与 `--data-root-hard` 指向同一预处理根即可。

**验证：**easy 目录若存在，同样运行 `verify_processed_dataset.py` 指向该路径。

**预期：**与阶段 1 相同契约。

---

### 阶段 3 — 单元测试（可选但推荐）

```bash
bash experiment/convergence_basin/run_unit_tests.sh -q
# 若收集阶段因 ROS pytest 插件报错，可先: export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
```

**预期：**`tests/` 通过或仅有与环境相关的 skip（例如未安装 open3d 时部分测试跳过）。

---

### 阶段 4 — 微调 SnowflakeNet

| 动作 | 命令示例 |
|------|----------|
| 默认微调（数据根指向 `PCN_far8_*`） | `python scripts/02_train_completion.py` |

训练脚本默认 **Group A（Encoder）lr=1e-5**、**Group B（Decoder）lr=5e-5**，并从 `ckpt-best-pcn-cd_l1.pth` 加载预训练。

**穿插验证 B — 损失仅在 Canonical 空间**

```bash
python experiment/convergence_basin/checks/verify_training_canonical_space.py \
  --data-root data/processed/PCN_far8_cano_in2048_gt16384 --split train --dry-run
```

**穿插验证 C — 训练前验证集 CD（脚本内 sanity）**

日志中出现 `[sanity] 训练前 val CD`，用于确认分布未错位。

| 检查项 | 预期效果（Go） |
|--------|----------------|
| 损失坐标系 | `CompletionDataset` 仅读 `{split}/input` 与 `{split}/gt`，均为 canonical；**不含 FPFH/`T_far` 位姿误差** |
| 预训练量级 | `val CD` 若在 PCN 上远高于论文量级（例如异常偏大），优先复查预处理与 `--ckpt_pretrain` 路径 |

---

### 阶段 5 — 评测补全（CD-L1×10³）

```bash
python scripts/03_eval_completion.py \
  --processed-root data/processed/PCN_far8_cano_in2048_gt16384 --split test \
  --ckpt checkpoints/snet_finetune/ckpt-best.pth
```

**穿插验证 D — 预训练 vs 微调**

```bash
bash experiment/convergence_basin/compare_cd_two_ckpts.sh \
  data/processed/PCN_far8_cano_in2048_gt16384 \
  Snet/SnowflakeNet-main/completion/checkpoints/ckpt-best-pcn-cd_l1.pth \
  checkpoints/snet_finetune/ckpt-best.pth
```

| 检查项 | 预期效果（Go） |
|--------|----------------|
| 微调 vs 预训练 | **微调**平均 CD-L1×10³ **低于**预训练；经验上常见 **相对降幅约 ≥20%**（依训练轮数与数据划分而定，**非硬性门禁**） |

---

### 阶段 6 — 反归一化数值约定（无数据亦可）

```bash
python experiment/convergence_basin/checks/verify_decanonicalize_row_major.py
```

**预期：**打印 `[OK]`，确认 `P_pred_w = (P_pred_cano * scale + centroid) @ R + t` 行向量约定与 `05_estimate_pose.py` 一致。

---

### 阶段 7 — 位姿闭环（FPFH + Gate-ICP + 最终 ICP）

```bash
python scripts/05_estimate_pose.py \
  --stem <样本 stem，无后缀> \
  --data-root data/processed/PCN_far8_cano_in2048_gt16384 \
  --ckpt checkpoints/snet_finetune/ckpt-best.pth \
  --gate-fitness 0.5
```

或使用双窗格可视化：

```bash
python scripts/07_viz_convergence_basin.py --stem <stem>
```

**穿插验证 E — Gate-ICP**

- 查看日志中 `gate_accepted` / `gate_fitness` vs `--gate-fitness`。
- **预期：**若 Gate 拒绝（fitness 过低），位姿初值回退 **`T_coarse`**，最终仍由 **原始 `obs_w` vs CAD** 的 ICP 收敛。

**穿插验证 F — FPFH 物理尺度**

- `--fpfh-voxel` 必须与 **点云物理单位**（米 / 毫米）一致；若 CAD 与观测量级错误，粗配会失败。可用 Open3D 查看 `complete.pcd` 包围盒尺度作为设定参考。

---

### 阶段 8 — Canonical 管线 sanity（可选）

```bash
PYTHONPATH=. python scripts/09_verify_cano_vs_legacy.py --pcn-root PCN --num-samples 5
```

**预期：**canonical 路径下 `pred` 半径与 CD 不出现「缩成一团」式坍缩（脚本说明中有细节）。

---

## 快速命令备忘

```bash
export PYTHONPATH="$(pwd):$(pwd)/Snet/SnowflakeNet-main:${PYTHONPATH:-}"
bash scripts/run_pipeline.sh          # 冒烟：打印推荐命令（环境变量见脚本内）
bash experiment/convergence_basin/run_unit_tests.sh -q
```

---

## 数据布局（`00` 之后）

```
data/processed/PCN_far8_cano_in2048_gt16384/{train,val,test}/{input,gt,obs_w,meta}/
```

`meta/*.npz` 典型字段：`C_cano`/`centroid_cano`、`scale_cano`、`view_idx`、`R_far`、`t_far`、`T_far_4x4`、`source_partial`、`source_complete` 等。逆变换工具：`src.data.preprocessing.apply_inverse_normalization`。

---

## 故障排查（摘要）

| 现象 | 建议 |
|------|------|
| Gate 始终拒绝 | 检查补全质量、`--gate-fitness`、`--fpfh-voxel`、以及反归一化是否与 meta 一致 |
| 微调 CD 不降 | 学习率组、是否加载预训练、`verify_processed_dataset` 是否全通过 |
| Cursor 里看不到 `src/data/` | `.cursorignore` 曾用 `data/` 会误匹配 `src/data/`；已改为仓库根的 `/data/`。若仍异常，以 `git ls-files src/data` 为准 |
| pytest 收集失败 | `export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`（见 `tests/conftest.py`） |
| Cursor / 终端卡死、反复 skip 后仍极慢 | ① 全局 pytest 插件（如 ROS）：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`。② 仅在 `tests/` 跑测试（`pytest.ini` 已 `testpaths=tests`）。③ 勿在仓库根堆放上千 `.npy`；`/debug/`、`/results/` 已忽略。④ Chamfer/CD 相关测试需要 PyTorch；环境缺 torch 会在收集阶段报错（并非无限 skip） |
| 疑惑「src 未被追踪」 | `.gitignore` 已为**黑名单**：默认跟踪 `src/`、`scripts/`、`tests/`、`experiment/` 源码；仅忽略数据集目录与大文件后缀。用 `git ls-files src` 核对 |
| `wandb`：`wandb_v1_*` 密钥 / 「API key must be 40 characters」 | **新版本密钥需要 wandb>=0.22.3 → Python≥3.8**。若在 Python 3.7 环境无法安装：`conda env create -f environment_wandb_online.yml` 新建 **Python≥3.10** 环境后再装 wandb；或 **`python scripts/02_train_completion.py --no-wandb`** |
| `scripts/`/`src/` 部分文件无法 `git add` 或推送后被忽略 | 多为「无前导 `/`」的目录规则误匹配子路径；现已收紧（`/PCN/`、`/wandb/` 等仅根目录）。自查：`git check-ignore -v -- <路径>`；确认为源码则核对文件名勿用大后缀匹配规则（如 `*.npz`） |
