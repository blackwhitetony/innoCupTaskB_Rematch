"""P2 稀疏基线 + P3 特征消融实验。

对应 plan.md 第 7/8/9 节。所有模型折内拟合，OOF 汇总后计算 pooled 指标。
结果追加写入 artifacts/metrics/runs.csv，并打印汇总。

用法：
    uv run python -m src.experiments            # 跑 P2 + P3
    uv run python -m src.experiments --stage p2 # 只跑 P2
    uv run python -m src.experiments --stage p3 # 只跑 P3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import HuberRegressor, LinearRegression, Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import T1_TARGET, T2_TARGET, T3_TARGET, build_frame
from .metrics import pooled_metrics
from .preprocessing import load_data

ROOT = Path(__file__).resolve().parents[1]
FOLDS_PATH = ROOT / "artifacts" / "folds" / "folds.csv"
RUNS_PATH = ROOT / "artifacts" / "metrics" / "runs.csv"
SEED_COL = "fold_seed42"


# ---------- 通用 OOF ----------

def oof_predict(X: np.ndarray, y: np.ndarray, fold: np.ndarray, fit_predict) -> np.ndarray:
    preds = np.full(len(y), np.nan)
    for k in np.unique(fold):
        tr = fold != k
        va = fold == k
        preds[va] = fit_predict(X[tr], y[tr])(X[va])
    return preds


def raw_fp(factory):
    def fp(Xtr, ytr):
        m = factory()
        m.fit(Xtr, ytr)
        return m.predict
    return fp


def pipe_fp(factory):
    def fp(Xtr, ytr):
        m = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("m", factory()),
        ])
        m.fit(Xtr, ytr)
        return m.predict
    return fp


def ratio_fp():
    def fp(Xtr, ytr):
        r = np.mean(ytr / Xtr[:, 0])
        return lambda Xv: Xv[:, 0] * r
    return fp


# ---------- T2 严格嵌套级联 ----------

def t2_cascade(df: pd.DataFrame, fold: np.ndarray, t1_model="ratio", inner_splits=5, seed=0):
    """严格嵌套 OOF：内层为 T1 生成 OOF 能耗，外层预测测试能耗，级联得到成本。

    返回 (energy_hat, cost_physics)。cost_physics = energy_hat * 电价。
    """
    weekly = df["weekly_travel_distance_km"].astype(float).values
    price = df["electricity_cost_per_kwh"].astype(float).values
    energy = df[T1_TARGET].astype(float).values

    def t1_fp(w_tr, y_tr):
        if t1_model == "ratio":
            r = np.mean(y_tr / w_tr)
            return lambda w: np.asarray(w) * r
        m = LinearRegression().fit(np.asarray(w_tr).reshape(-1, 1), y_tr)
        return lambda w: m.predict(np.asarray(w).reshape(-1, 1))

    energy_hat = np.full(len(df), np.nan)
    for k in np.unique(fold):
        A = fold != k
        B = fold == k
        idx_A = np.where(A)[0]
        # 内层：为 A 生成 OOF 能耗
        kf = KFold(n_splits=inner_splits, shuffle=True, random_state=seed)
        for tr_i, va_i in kf.split(idx_A):
            tr_idx = idx_A[tr_i]
            va_idx = idx_A[va_i]
            energy_hat[va_idx] = t1_fp(weekly[tr_idx], energy[tr_idx])(weekly[va_idx])
        # 外层：A 全量拟合，预测 B
        energy_hat[B] = t1_fp(weekly[A], energy[A])(weekly[B])
    return energy_hat, energy_hat * price


# ---------- 运行记录 ----------

def record(task: str, model: str, feature_set: str, y: np.ndarray, pred: np.ndarray, p: int, notes: str = "") -> dict:
    m = pooled_metrics(y, pred, p)
    m.update({"task": task, "model": model, "feature_set": feature_set, "notes": notes})
    return m


# ---------- P2 基线 ----------

def run_p2(df: pd.DataFrame, fold: np.ndarray, runs: list[dict]) -> None:
    frame, blocks = build_frame(df)

    # T1：周里程 三个公式/线性基线
    y1 = df[T1_TARGET].astype(float).values
    X1 = frame[["weekly_travel_distance_km"]].astype(float).values
    runs.append(record("T1", "ratio_mean", "F0_core", y1,
                       oof_predict(X1, y1, fold, ratio_fp()), 1))
    runs.append(record("T1", "ols_origin", "F0_core", y1,
                       oof_predict(X1, y1, fold, raw_fp(lambda: LinearRegression(fit_intercept=False))), 1))
    runs.append(record("T1", "ols", "F0_core", y1,
                       oof_predict(X1, y1, fold, raw_fp(lambda: LinearRegression())), 1))

    # T3：日通勤 OLS / 过原点 / Huber
    y3 = df[T3_TARGET].astype(float).values
    X3 = frame[["daily_commute_km"]].astype(float).values
    runs.append(record("T3", "ols", "F0_core", y3,
                       oof_predict(X3, y3, fold, raw_fp(lambda: LinearRegression())), 1))
    runs.append(record("T3", "ols_origin", "F0_core", y3,
                       oof_predict(X3, y3, fold, raw_fp(lambda: LinearRegression(fit_intercept=False))), 1))
    runs.append(record("T3", "huber", "F0_core", y3,
                       oof_predict(X3, y3, fold, raw_fp(lambda: HuberRegressor())), 1))

    # T2：严格级联 / Oracle / 无级联基线
    y2 = df[T2_TARGET].astype(float).values
    price = df["electricity_cost_per_kwh"].astype(float).values
    energy_true = df[T1_TARGET].astype(float).values
    _, cost_physics = t2_cascade(df, fold, t1_model="ratio")
    runs.append(record("T2", "cascade_cost_physics", "strict_nested_oof", y2, cost_physics, 2,
                       "energy_hat=ratio_mean(weekly); cost=energy_hat*price"))
    runs.append(record("T2", "oracle_cost_physics", "oracle", y2, energy_true * price, 2,
                       "使用真实能耗，仅作上界"))
    X2_nc = frame[["weekly_travel_distance_km", "daily_commute_km", "electricity_cost_per_kwh"]].astype(float).values
    runs.append(record("T2", "no_cascade_ridge", "weekly+daily+price", y2,
                       oof_predict(X2_nc, y2, fold, pipe_fp(lambda: Ridge())), X2_nc.shape[1]))


# ---------- P3 消融 ----------

def _lin_oof(frame, blocks, task, keys, y, fold, model_name, model_factory, runs, notes=""):
    cols = []
    seen = set()
    for k in keys:
        for c in blocks[k]:
            if c not in seen:
                seen.add(c); cols.append(c)
    X = frame[cols].astype(float).values
    pred = oof_predict(X, y, fold, pipe_fp(model_factory))
    runs.append(record(task, model_name, "+".join(keys), y, pred, X.shape[1], notes))


def run_p3(df: pd.DataFrame, fold: np.ndarray, runs: list[dict]) -> None:
    frame, blocks = build_frame(df)
    b1, b3 = blocks["T1"], blocks["T3"]
    y1 = df[T1_TARGET].astype(float).values
    y3 = df[T3_TARGET].astype(float).values

    # --- T1 特征块消融（OLS） ---
    t1_combos = [
        ["F0_core", "F1_trip"],
        ["F0_core", "F2_official"],
        ["F0_core", "F3_vehicle_region"],
        ["F0_core", "F4_other"],
        ["F0_core", "F1_trip", "F2_official", "F3_vehicle_region", "F4_other"],
        ["F1_alt_monthly"],  # 周里程替换为估算月里程
    ]
    for combo in t1_combos:
        _lin_oof(frame, b1, "T1", combo, y1, fold, "ols", lambda: LinearRegression(), runs)

    # --- T3 特征块消融（OLS） ---
    t3_combos = [
        ["F0_core", "F1_trip"],
        ["F0_core", "F2_official"],
        ["F0_core", "F3_vehicle_region"],
        ["F0_core", "F4_other"],
        ["F0_core", "F1_trip", "F2_official", "F3_vehicle_region", "F4_other"],
    ]
    for combo in t3_combos:
        _lin_oof(frame, b3, "T3", combo, y3, fold, "ols", lambda: LinearRegression(), runs)

    # --- T3 负值策略消融 ---
    # winsor 目标（clip 到 [0, 99分位]，评估仍用原 y）
    q99 = float(np.percentile(y3, 99))
    y3_win = np.clip(y3, 0, q99)
    X3 = frame[["daily_commute_km"]].astype(float).values
    runs.append(record("T3", "ols_winsor_target", "F0_core", y3,
                       oof_predict(X3, y3_win, fold, raw_fp(lambda: LinearRegression())), 1,
                       f"训练目标 clip 到 [0,{q99:.0f}]，评估用原 y"))

    # delete negatives：每个训练折剔除 y<0，预测全部验证折
    pred_del = np.full(len(y3), np.nan)
    for k in np.unique(fold):
        tr = (fold != k) & (y3 >= 0)
        va = fold == k
        m = LinearRegression().fit(X3[tr], y3[tr])
        pred_del[va] = m.predict(X3[va])
    runs.append(record("T3", "ols_delete_neg_train", "F0_core", y3, pred_del, 1,
                       "训练折剔除负目标样本，预测全部"))

    # --- T2 消融：级联 + 线性校准 / 级联 + 燃油支出 ---
    y2 = df[T2_TARGET].astype(float).values
    energy_hat, cost_physics = t2_cascade(df, fold, t1_model="ratio")
    calib_cols = ["cost_physics", "energy_hat", "electricity_cost_per_kwh"]
    calib_frame = pd.DataFrame({
        "cost_physics": cost_physics,
        "energy_hat": energy_hat,
        "electricity_cost_per_kwh": df["electricity_cost_per_kwh"].astype(float),
    })
    Xc = calib_frame[calib_cols].astype(float).values
    runs.append(record("T2", "cascade_calibration_ols", "cost_physics+energy_hat+price", y2,
                       oof_predict(Xc, y2, fold, pipe_fp(lambda: LinearRegression())), Xc.shape[1],
                       "在 cost_physics 上做线性校准"))

    Xf = pd.concat([calib_frame, df[["fuel_expense_per_month"]].astype(float)], axis=1).values
    runs.append(record("T2", "cascade_fuel_feature", "cost_physics+fuel", y2,
                       oof_predict(Xf, y2, fold, pipe_fp(lambda: Ridge())), Xf.shape[1],
                       "官方建议变量 fuel_expense 入模（需确认测试可得）"))


# ---------- 汇总 ----------

def save_runs(runs: list[dict]) -> None:
    df = pd.DataFrame(runs)
    cols = ["task", "model", "feature_set", "p", "n", "r2", "adj_r2", "rmse", "mae", "notes"]
    df = df[cols]
    RUNS_PATH.parent.mkdir(parents=True, exist_ok=True)
    header = not RUNS_PATH.exists()
    df.to_csv(RUNS_PATH, mode="a", header=header, index=False)
    return df


def print_summary(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    show = df[["task", "model", "feature_set", "p", "r2", "adj_r2", "rmse", "mae"]]
    print(show.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=["p2", "p3", "all"])
    ap.add_argument("--csv", default=str(ROOT / "复赛B题" / "B题数据集.csv"))
    args = ap.parse_args()

    df = load_data(args.csv)
    folds = pd.read_csv(FOLDS_PATH)
    fold = folds[SEED_COL].values

    runs: list[dict] = []
    if args.stage in ("p2", "all"):
        run_p2(df, fold, runs)
    if args.stage in ("p3", "all"):
        run_p3(df, fold, runs)

    result = save_runs(runs)
    print_summary(result)
    print(f"\n已写入 {RUNS_PATH}（追加）")


if __name__ == "__main__":
    main()
