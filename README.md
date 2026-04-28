# SnowflakeNet + FPFH/ICP — PCN 最简流水线

## 主仓库

- 编号脚本（`scripts/`）:
  1. `00_preprocess_pcn.py` — PCN 原始 .pcd → 随机 `T_far` + bbox/PCA → `input/`, `gt/`, `obs_w/`, `meta/`
  2. `01_compute_best_view.py` — 每模型 HPR 最优 partial 视角 → `data/processed/PCN_hpr_best_views.json`
  3. `02_train_completion.py` — SnowflakeNet 微调（仅训练 + mini-val，完整 CD 用 `03`）
  4. `03_eval_completion.py` — 预处理 .npy 上 CD-L1×10³
  5. `04_infer_completion.py` — 单文件 `.pcd` 补全（可选 `01` 的 JSON）
  6. `05_estimate_pose.py` — 补全 + 逆归一化 + 可选 FPFH 粗配 + ICP，输出 `T_icp`（对 `obs_w`）

- 公共代码在 `src/`（`data/preprocessing`, `data/pcn_dataset`, `models/chamfer`, `models/snet_loader`, `pose_estimation/postprocess`, `evaluation/`, `inference/` 等）。

- 需本地克隆 **SnowflakeNet** 到 `Snet/SnowflakeNet-main/`（`models`、`loss_functions/Chamfer3D` 等），与 `.gitignore` 中忽略规则一致。

- **Legacy**：旧版 ITODD/ModelNet/消融等脚本在姊妹仓 [`../SnowflakeNet_FPFH_ICP_legacy`](../SnowflakeNet_FPFH_ICP_legacy)（如路径不同请自行替换）。

## 数据布局（`00` 之后）

`data/processed/PCN_far_in2048_gt16384/{split}/{input,gt,obs_w,meta}/`

## 快速串行

```bash
bash scripts/run_pipeline.sh   # 见脚本内环境变量
```
