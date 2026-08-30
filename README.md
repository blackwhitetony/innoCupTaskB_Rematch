# innoCupTaskB_Rematch

2026 年第五届"创新杯"大学生大数据挑战赛 **复赛 B 题**——从燃油到纯电：电动汽车消费者转型分析。

## 题目与任务

基于 50,000 条电动汽车消费者记录，完成三个回归任务（评分指标为调整决定系数 adj-R²）：

| 任务 | 预测目标 | 最终模型 | adj-R² |
|---|---|---|---|
| T1 月度能耗预测 | `monthly_energy_consumption_kwh` | `0.8163487 × weekly_travel_distance_km` | 0.876298 |
| T2 月度充电成本预测 | `monthly_charging_cost` | `0.8163487 × weekly × electricity_cost_per_kwh` | 0.917817 |
| T3 燃油替代支出预测 | `fuel_expense_per_month` | `8.3947643 × daily_commute_km` | 0.905416 |

核心结论：数据由近似确定性的物理关系生成，稀疏闭式公式优于复杂模型（浅层 LightGBM 残差模型 Bootstrap 置信区间全面为负）。

## 环境

- Python 3.11，依赖管理使用 [uv](https://docs.astral.sh/uv/)
- 仅需 CPU，无需 GPU

```bash
uv sync
```

## 复现流程

```bash
# 1. 数据校验（需先把 B题数据集.csv 放到 复赛B题/ 目录）
uv run python -m src.schema_check

# 2. 生成固定折划分
uv run python -m src.folds

# 3. P2 稀疏基线 + P3 特征消融
uv run python -m src.experiments --stage all

# 4. P4 树模型残差确认 + P6 5折×3种子最终确认
uv run python -m src.finalize

# 5. 拟合最终模型 + 生成提交 CSV
uv run python -m src.submit            # 用当前数据生成示例提交
uv run python -m src.submit --test 测试.csv   # 对正式测试集预测
```

## 目录结构

```
configs/     数据 Schema 与泄露边界（schema.yaml / leakage.yaml）
src/         代码（schema 校验、预处理、特征、基线/消融、最终确认、提交）
autodl/      在线平台复现脚本（setup.sh / run.sh）
paper/       论文 LaTeX 源码（main.tex + chapters/ + figures/）
reports/     结果摘要与方法论笔记
```

## 说明

- 数据 CSV 不随仓库提交，需自行从赛题目录复制到 `复赛B题/B题数据集.csv`。
- 关键口径（泄露边界、adj-R² 的 `p` 定义、测试集字段）待组委会确认，详见 `configs/leakage.yaml` 与论文第 3 节。
