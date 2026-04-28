# SnowflakeNet + FPFH/ICP — PCN 最简流水线

## 主仓库

- 编号脚本（`scripts/`）:
  1. `00_preprocess_pcn.py` — PCN 原始 .pcd → **按 complete 的 AABB 中心 + max-radius 单位球** 归一化得到 canonical `input/`, `gt/`（与 PCN 预训练分布一致，**不做 PCA**）；随机 `T_far` 仅作用于 `obs_w/`；`meta/` 同时存 canonical (`C_cano`, `scale_cano`, 可选 `R_aug`) 与世界刚体 (`R_far`, `t_far`, `T_far_4x4`)。`--mode easy` 仅保留每模型 HPR 最佳视角用于 curriculum；`--rot-aug-deg`/`--rot-aug-axis` 可在 canonical 系内做绕重力轴的小角度扰动。**并行单元为每个 model**：worker 在内部仅读 `complete.pcd` 一次，再循环处理该 model 的所有 partial（hard 模式 ~8×），显著降低重复解码与随机读。
  2. `01_compute_best_view.py` — 每模型 HPR 最优 partial 视角 → `data/processed/PCN_hpr_best_views.json`（HPR 已在 canonical 系评估，无 PCA）。同一模型内 **只读一次** `complete.pcd`。`--workers`：本地 NVMe 可适当调高；**共享盘 / 云容器盘** 常见甜点在 **4～8**，过高会因随机读拥塞反而变慢（可用 `iostat` 看 `%util` / `await`）。`--timing-stats` 打印每模型耗时分位数，便于回归。
  3. `02_train_completion.py` — SnowflakeNet 微调（支持 easy→hard curriculum；完整 CD 用 `03`）
  4. `03_eval_completion.py` — 预处理 .npy 上 CD-L1×10³
  5. `04_infer_completion.py` — 单文件 `.pcd` 补全（可选 `01` 的 JSON）
  6. `05_estimate_pose.py` — 补全 + 逆归一化 + 可选 FPFH 粗配 + ICP，输出 `T_icp`（对 `obs_w`）
  7. `06_viz_lighting_dropout.py` — 交互式可视化 `transforms` 中强背光/强反光 dropout，Open3D 场景光源与参数滑条
  8. `08_viz_completion_pretrain_vs_finetune.py` — 预处理 `.npy` 上对比预训练与微调 ckpt 的 Open3D 可视化（可选 `--pretrain-ablation`）
  9. `09_verify_cano_vs_legacy.py` — **极少量** PCN 样本：旧 bbox+PCA 与 **新 canonical（按 complete）** 归一化下预训练 SnowflakeNet 的 `pred_max` / CD-L1，快速确认 canonical 管线无「缩成一团」坍缩（`PYTHONPATH=. python scripts/09_verify_cano_vs_legacy.py --num-samples 5`）

- 公共代码在 `src/`（`data/preprocessing`, `data/pcn_dataset`, `models/chamfer`, `models/snet_loader`, `pose_estimation/postprocess`, `evaluation/`, `inference/` 等）。

- 需本地克隆 **SnowflakeNet** 到 `Snet/SnowflakeNet-main/`（`models`、`loss_functions/Chamfer3D` 等），与 `.gitignore` 中忽略规则一致。

- **Legacy**：旧版 ITODD/ModelNet/消融等脚本在姊妹仓 [`../SnowflakeNet_FPFH_ICP_legacy`](../SnowflakeNet_FPFH_ICP_legacy)（如路径不同请自行替换）。

## 数据布局（`00` 之后）

`data/processed/PCN_far_cano_in2048_gt16384/{split}/{input,gt,obs_w,meta}/`

curriculum easy 集：

`data/processed/PCN_hpr_easy_cano_in2048_gt16384/{split}/{input,gt,obs_w,meta}/`

`meta/*.npz` 字段（新 schema）：`C_cano (3,)`、`scale_cano ()`、`R_aug (3,3)`（默认单位阵）、
`R_far (3,3)`、`t_far (3,)`、`T_far_4x4 (4,4)`、`split/synset/model_id/view_idx/source_*`。
`src.data.preprocessing.apply_inverse_normalization` 自动识别新/旧 schema。

> Legacy：旧目录名 `PCN_far_in*` / `PCN_hpr_easy_in*` 与旧 meta 字段
> `C_bbox/scale/R_pca/mu_pca` 仍可被 `apply_inverse_normalization` 兼容读取。

生成顺序：

```bash
# 共享盘/容器盘建议 --workers 4~8；本地 NVMe 可调高。诊断可加 01 的 --timing-stats。
python scripts/01_compute_best_view.py --pcn-root PCN --splits train,val,test --workers 8
python scripts/00_preprocess_pcn.py --pcn-root PCN --mode easy --splits train,val,test --workers 8
python scripts/00_preprocess_pcn.py --pcn-root PCN --mode hard --splits train,val,test --workers 8
python scripts/02_train_completion.py \
  --data-root-easy data/processed/PCN_hpr_easy_cano_in2048_gt16384 \
  --data-root-hard data/processed/PCN_far_cano_in2048_gt16384
```

## 快速串行

```bash
bash scripts/run_pipeline.sh   # 见脚本内环境变量
```
