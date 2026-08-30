"""特征矩阵构建与特征块定义（P0 泄露边界 + P3 特征消融共用）。

对应 plan.md 第 4 节（泄露边界）与第 7/9 节（特征块）。

- build_frame 生成统一特征矩阵（数值列保留 NaN，由折内 Pipeline 填补；
  类别列用常量 "Missing" 填补后 One-Hot，不涉及数据学习，可全局执行）。
- BLOCKS 定义各任务特征块（列名列表），供消融实验按块加入。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .preprocessing import load_schema, get_column_sets

# 类别列与数值/二值列（来自 schema）
_CS = get_column_sets()
CATEGORICAL = _CS["categorical"]
BINARY = _CS["binary"]
NUMERIC = _CS["numeric"]

# 各任务目标列
TARGETS = load_schema()["targets"]
T1_TARGET = TARGETS["T1_energy"]
T2_TARGET = TARGETS["T2_cost"]
T3_TARGET = TARGETS["T3_fuel"]


def _one_hot(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = df[col].astype(str).str.strip().replace({"nan": "Missing"})
    return pd.get_dummies(s, prefix=col, drop_first=True, dtype=np.int8)


def build_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """返回 (frame, blocks)。frame 列均为命名特征，数值列可能含 NaN。"""
    out = pd.DataFrame(index=df.index)

    # 数值列（原始，保留 NaN）
    for c in NUMERIC:
        out[c] = df[c].astype(float)

    # 二值列
    for c in BINARY:
        out[c] = df[c].astype(float)

    # 类别 One-Hot（drop first）
    cat_cols: list[str] = []
    for c in CATEGORICAL:
        oh = _one_hot(df, c)
        out = pd.concat([out, oh], axis=1)
        cat_cols += list(oh.columns)

    # 派生特征
    daily = df["daily_commute_km"].astype(float)
    weekly = df["weekly_travel_distance_km"].astype(float)
    out["monthly_distance"] = weekly * 52.0 / 12.0
    out["weekly_daily_ratio"] = weekly / daily.replace(0, np.nan)
    out["non_commute_km"] = weekly - 5.0 * daily
    out["log_income"] = np.log1p(df["annual_income"].astype(float))

    # 官方交互（车型 3 类 one-hot × 里程；城市 2 类 one-hot × 日通勤）
    vehicle_oh = [c for c in cat_cols if c.startswith("current_vehicle_type_")]
    city_oh = [c for c in cat_cols if c.startswith("city_type_")]
    for v in vehicle_oh:
        out[f"weekly_x_{v}"] = weekly * out[v]
        out[f"daily_x_{v}"] = daily * out[v]
    for c in city_oh:
        out[f"daily_x_{c}"] = daily * out[c]

    # 特征块定义
    weekly_blocks = {
        "F0_core": ["weekly_travel_distance_km"],
        "F1_trip": ["daily_commute_km", "weekly_daily_ratio", "non_commute_km"],
        "F1_alt_monthly": ["monthly_distance"],  # 与 F0 互斥，作为周里程替代
        "F2_official": [f"weekly_x_{v}" for v in vehicle_oh] + [f"daily_x_{v}" for v in vehicle_oh],
        "F3_vehicle_region": vehicle_oh + city_oh + ["vehicle_age_years"],
        "F4_other": (
            ["age", "annual_income", "log_income", "charging_station_accessibility",
             "nearest_charging_station_km", "electricity_cost_per_kwh",
             "environmental_awareness_score", "government_incentive_awareness",
             "technology_affinity_score", "range_anxiety_score",
             "battery_replacement_concern", "ev_knowledge_score"]
            + BINARY
            + [c for c in cat_cols if c.startswith("education_level_")
               or c.startswith("ev_adoption_likelihood_")]
        ),
    }

    daily_blocks = {
        "F0_core": ["daily_commute_km"],
        "F1_trip": ["weekly_travel_distance_km", "weekly_daily_ratio", "non_commute_km"],
        "F2_official": [f"daily_x_{v}" for v in vehicle_oh] + [f"daily_x_{c}" for c in city_oh],
        "F3_vehicle_region": vehicle_oh + city_oh + ["vehicle_age_years", "log_income"],
        "F4_other": (
            ["age", "annual_income", "charging_station_accessibility",
             "nearest_charging_station_km", "environmental_awareness_score",
             "government_incentive_awareness", "technology_affinity_score",
             "range_anxiety_score", "battery_replacement_concern", "ev_knowledge_score"]
            + BINARY
            + [c for c in cat_cols if c.startswith("education_level_")
               or c.startswith("ev_adoption_likelihood_")]
        ),
    }

    blocks = {"T1": weekly_blocks, "T3": daily_blocks}
    return out, blocks


def select_features(frame: pd.DataFrame, blocks: dict, keys: list[str]) -> list[str]:
    """按块名取并集去重，保持传入顺序。"""
    cols: list[str] = []
    seen: set[str] = set()
    for k in keys:
        for c in blocks[k]:
            if c not in seen:
                seen.add(c)
                cols.append(c)
    return [c for c in cols if c in frame.columns]
