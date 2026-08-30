# 复赛 B 题速查（选题：B —— 从燃油到纯电：电动汽车消费者转型分析）

> 来源：`复赛B题/复赛B：从燃油到纯电：电动汽车消费者转型分析.pdf` + `复赛B题/B题数据集.csv` 实测
> 最后更新：2026-08-30

## 1. 一句话概括
基于 50,000 条电动汽车消费者记录（人口统计/通勤/充电设施/环保意识/技术接受度/财务），完成 **3 个回归任务**，评分依据 = 组委会测试集上的 **调整决定系数 (adj-R²)**。

## 2. 三个任务（全部为回归）

| 任务 | 目标列 | 官方难点提示 | 要求提交 |
|---|---|---|---|
| T1 月度能耗预测 | `monthly_energy_consumption_kwh` | 与行驶距离强相关；需挖掘 `daily_commute_km` × `current_vehicle_type` 交互特征 | 特征工程报告、预测结果、adj-R²₁ |
| T2 月度充电成本预测 | `monthly_charging_cost` | 高度非线性；处理异常值（极高收入/极长通勤）；建议特征：`electricity_cost_per_kwh`、`monthly_energy_consumption_kwh`（可用 T1 预测值级联）、`fuel_expense_per_month`、`government_incentive_awareness` | 同上，adj-R²₂ |
| T3 燃油替代支出预测（"反事实"） | `fuel_expense_per_month` | 官方称最难；"车型+里程+地域"→油价映射；训练时把该列视为真值，测试集上预测 | 同上，adj-R²₃ |

## 3. 提交说明（原文要点）
1. 每任务提交：模型代码、预测结果 CSV（含样本行号、真值、预测值）、建模报告（特征重要性、明显指向性/强相关/泄露特征剔除思路、模型选择依据、调参过程）。
2. 排名依据：组委会测试集上的 (调整)R²；当前目录未见独立测试集，正式字段与提交口径需确认。
3. 自行划分训练/验证集；**禁止用测试集真值**训练或选参。

## 4. 数据集实测概况（B题数据集.csv）
- 形状：50000 × 23；无重复行；`education_level`、`charging_station_accessibility`、`ev_knowledge_score` 各缺失 500 条（各 1%）。
- 数值列 19 个，object 列 4 个：
  - `education_level`: High School / Bachelor / Master / PhD（可序数编码）
  - `city_type`: Urban / Suburban / Rural
  - `current_vehicle_type`: Hatchback / Sedan / SUV / Truck
  - `ev_adoption_likelihood`: Low / Medium / High（可序数编码）
- 列清单：age, annual_income, education_level, city_type, daily_commute_km, weekly_travel_distance_km, current_vehicle_type, vehicle_age_years, fuel_expense_per_month, charging_station_accessibility, nearest_charging_station_km, home_charging_available, electricity_cost_per_kwh, environmental_awareness_score, government_incentive_awareness, technology_affinity_score, range_anxiety_score, battery_replacement_concern, ev_knowledge_score, previous_ev_experience, monthly_energy_consumption_kwh, monthly_charging_cost

### 异常值备忘
- **`fuel_expense_per_month` 有 271 个负值（min=-99.7）** —— 业务上异常，但当前实测删除或截断会降低 OOF R²；评分模型默认保留且不取对数。
- `annual_income` 尾部极高（max 250k，99.9%≈195k）；`weekly_travel_distance_km` 到 732。
- 各评分列均为 1–10 浮点；`electricity_cost_per_kwh` 范围 0.08–0.35，共 28 个离散值。

## 5. 已验证实证结论（本仓库基线）
- 裸 HistGradientBoosting（100 iter、4000 行、3 折 CV）R²：T1≈0.86，T2≈0.90（不含能耗特征），T3≈0.90。
- **确定性关系**：`monthly_charging_cost ≈ monthly_energy_consumption_kwh × electricity_cost_per_kwh`，全量数据 R²=0.9994 → T2 近似白送（用 T1 预测值级联或直接公式），但报告需按题目口径写成"级联特征"。
- T2 能否直接使用真实 `monthly_energy_consumption_kwh` 取决于测试集是否提供该字段；在确认前以 **T1 的严格 OOF 预测值** 作级联特征，真实值只作 Oracle 上界，避免训练/推理不一致。
- T3 默认排除 `monthly_energy_consumption_kwh` 与 `monthly_charging_cost` 等后置结果；`fuel_expense_per_month` 与通勤里程强相关，合法特征基线已到约 0.905。

## 6. 分工建议（待讨论）
- 数据清洗 + EDA + 特征工程（交互特征：commute×vehicle_type、travel×city_type 等）
- 模型：先做公式/OLS/Ridge 基线；仅在残差存在稳定信号时使用浅层 CatBoost/LightGBM/XGBoost 或 OOF 融合
- 报告：按提交要求组织（特征重要性、泄露剔除、调参过程）
- 划分策略：K-Fold CV 上报 adj-R²；注意 adj-R² 需在固定特征集下按 `1-(1-R²)(n-1)/(n-p-1)` 计算或说明口径
