"""P7 全量重训 + 提交 CSV 生成。

三个最终模型均为闭式公式（在全量训练数据上拟合）：
- T1：energy_hat = r1 * weekly_travel_distance_km，r1 = mean(energy / weekly)
- T3：fuel_hat = beta * daily_commute_km，beta 为过原点 OLS 系数
- T2：cost_hat = r1 * weekly * electricity_cost_per_kwh（复用 T1 系数）

用法：
    uv run python -m src.submit                    # 拟合最终模型 + 用当前数据生成示例提交
    uv run python -m src.submit --test 测试.csv     # 对正式测试集预测并生成提交 CSV
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import load_data
from .features import T1_TARGET, T2_TARGET, T3_TARGET

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts" / "models"
SUBMIT_DIR = ROOT / "submissions"


def fit_final_models(df: pd.DataFrame) -> dict:
    weekly = df["weekly_travel_distance_km"].astype(float).values
    daily = df["daily_commute_km"].astype(float).values
    energy = df[T1_TARGET].astype(float).values
    fuel = df[T3_TARGET].astype(float).values

    r1 = float(np.mean(energy / weekly))
    beta = float(np.sum(daily * fuel) / np.sum(daily * daily))
    return {"T1": {"r1": r1, "formula": "energy = r1 * weekly_travel_distance_km"},
            "T3": {"beta": beta, "formula": "fuel = beta * daily_commute_km"},
            "T2": {"formula": "cost = r1 * weekly * electricity_cost_per_kwh", "depends_on": "T1.r1"}}


def predict(df: pd.DataFrame, models: dict) -> dict:
    weekly = df["weekly_travel_distance_km"].astype(float).values
    daily = df["daily_commute_km"].astype(float).values
    price = df["electricity_cost_per_kwh"].astype(float).values
    r1 = models["T1"]["r1"]
    beta = models["T3"]["beta"]
    return {
        T1_TARGET: r1 * weekly,
        T2_TARGET: r1 * weekly * price,
        T3_TARGET: beta * daily,
    }


def write_submission(df: pd.DataFrame, preds: dict, models: dict, tag: str) -> list[Path]:
    SUBMIT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for target, formula_key in [(T1_TARGET, "T1"), (T2_TARGET, "T2"), (T3_TARGET, "T3")]:
        out = pd.DataFrame({
            "row_id": np.arange(len(df)),
            "true": df[target].astype(float).values if target in df.columns else np.nan,
            "prediction": np.round(preds[target], 6),
        })
        p = SUBMIT_DIR / f"{formula_key}_{tag}.csv"
        out.to_csv(p, index=False)
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "复赛B题" / "B题数据集.csv"))
    ap.add_argument("--test", default=None)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    df = load_data(args.csv)
    models = fit_final_models(df)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "final_models.json").write_text(
        json.dumps(models, ensure_ascii=False, indent=2), encoding="utf-8")

    print("最终模型系数：")
    print(json.dumps(models, ensure_ascii=False, indent=2))

    if args.test:
        test = load_data(args.test)
        preds = predict(test, models)
        paths = write_submission(test, preds, models, "submission")
        print("\n提交文件：")
    else:
        preds = predict(df, models)
        paths = write_submission(df, preds, models, "demo_in_sample")
        print("\n示例提交（当前数据，含真值列，供格式核对）：")

    for p in paths:
        print(" ", p)


if __name__ == "__main__":
    main()
