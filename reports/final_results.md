# B 题最终结果摘要（供论文撰写）

> 数据：`复赛B题/B题数据集.csv`（50,000×23，SHA-256 见 configs/schema.yaml）
> 验证口径：5 折 shuffled CV × 3 种子（42/2026/7），fold_id 见 artifacts/folds/folds.csv

## 1. 冻结的最终模型（闭式公式，全量数据拟合）

| 任务 | 目标 | 最终公式 | 系数 |
|---|---|---|---|
| T1 | `monthly_energy_consumption_kwh` | `energy = r1 × weekly_travel_distance_km` | r1 = 0.8163487 |
| T2 | `monthly_charging_cost` | `cost = r1 × weekly × electricity_cost_per_kwh` | 复用 T1.r1 |
| T3 | `fuel_expense_per_month` | `fuel = β × daily_commute_km` | β = 8.3947643 |

模型参数存档：`artifacts/models/final_models.json`

## 2. 最终指标（5 折 × 3 种子 pooled OOF）

| 任务 | adj-R² 均值 | 标准差 | RMSE | MAE |
|---|---:|---:|---:|---:|
| T1 | 0.876298 | 0.000007 | 31.04 | 24.59 |
| T2 | 0.917817 | 0.000004 | 7.15 | 5.32 |
| T3 | 0.905416 | 0.000002 | 40.03 | 31.96 |

- 样本内 R² 与 OOF 差距 < 1e-4，确认无过拟合（模型即数据生成公式）。

## 3. 关键消融结论（论文可用）

1. **稀疏物理公式优于复杂模型**：三个任务的浅层 LightGBM 残差模型相对基线的
   Bootstrap 95% ΔR² 区间整体小于 0（T1 -0.001430、T3 -0.000355、T2 -0.000922），
   即复杂模型无一通过晋级门槛。
2. **题面交互项无增益**：T1 的 `里程×车型`、T3 的 `里程×车型/城市` 交互加入后
   adj-R² 均下降。
3. **异常值保留**：T3 负燃油样本保留原值最优；winsor / 删除负样本 / Huber 均略降。
4. **T2 的确定性关系**：`cost ≈ energy × 电价`（Oracle R²=0.9994）。正式提交采用
   严格嵌套 OOF 的 T1 预测作级联，避免 teacher forcing；真实能耗仅作上界对照。

## 4. 提交文件（submissions/）

- 格式：`row_id, true, prediction`（真值列在盲测提交时由组委会填充）
- 当前示例：`T{1,2,3}_demo_in_sample.csv`（含真值，供格式核对）
- 正式测试集到达后执行：`uv run python -m src.submit --test <测试.csv>`

## 5. 待组委会确认项

- adj-R² 中 `p` 的统一定义（One-Hot / 公式级联模型的自由度）
- 测试集是否隐藏三个目标列；T2 是否提供真实能耗/燃油
- 预测 CSV 的真值列在盲测时如何填写
