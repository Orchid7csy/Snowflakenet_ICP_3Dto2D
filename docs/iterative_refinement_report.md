# Iterative Refinement 记录说明

## 建议
- 默认按 `fitness` 选择最优迭代（`--iter-select best_fitness`），与 Gate 判定一致。
- 若业务更关注局部几何误差，可尝试 `--iter-select best_rmse` 对比。

## 启用多次迭代
```bash
python scripts/05_estimate_pose.py --eval-all --iterative-refine --max-iter 3
```

## 迭代上限
- 参数：`--max-iter`。
- 默认：`3`。
- 建议先在 `3~5` 范围网格测试，再按类别统计收敛收益。

## 导出每轮提升报告
```bash
python scripts/05_estimate_pose.py \
  --eval-all --iterative-refine --max-iter 3 \
  --iter-select best_fitness \
  --iter-report-csv outputs/iter_report.csv
```

CSV 字段：
- `stem`, `class`, `iter_idx`
- `fitness`, `rmse`
- `delta_fitness`（相对上一轮增量）
- `delta_rmse`（相对上一轮下降量，正值表示变好）
