"""P4 树模型残差确认 + P6 5 折×3 种子最终确认。

对应 plan.md 第 6.3 节晋级门槛与第 12 阶段 P4/P6：
- P4：浅层 LightGBM 拟合稀疏基线的折内残差，看是否越过晋级门槛（ΔR²≥0.0002）。
- P6：对三任务最优基线做 5 折×3 种子确认，报 pooled adj-R² 的均值与标准差。
本地与 AutoDL 均可运行，仅需 CPU。

用法：
    uv run python -m src.finalize [--csv 复赛B题/B题数据集.csv]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression

from .features import T1_TARGET, T2_TARGET, T3_TARGET, build_frame
from .metrics import r2, pooled_metrics
from .preprocessing import load_data
from .experiments import t2_cascade

ROOT = Path(__file__).resolve().parents[1]
FOLDS_PATH = ROOT / "artifacts" / "folds" / "folds.csv"
OUT_PATH = ROOT / "artifacts" / "metrics" / "finalize_runs.csv"
SEEDS = [42, 2026, 7]


def ratio_base():
    def fp(Xtr, ytr):
        r = np.mean(ytr / Xtr[:, 0])
        return lambda Xv: Xv[:, 0] * r
    return fp


def origin_ols_base():
    def fp(Xtr, ytr):
        m = LinearRegression(fit_intercept=False).fit(Xtr, ytr)
        return m.predict
    return fp


def residual_boost_oof(X_base, X_tree, y, fold, base_fp, tree_factory):
    final = np.full(len(y), np.nan)
    for k in np.unique(fold):
        tr = fold != k
        va = fold == k
        pred_tr = base_fp(X_base[tr], y[tr])(X_base[tr])
        pred_va = base_fp(X_base[tr], y[tr])(X_base[va])
        r_tr = y[tr] - pred_tr
        t = tree_factory()
        t.fit(X_tree[tr], r_tr)
        final[va] = pred_va + t.predict(X_tree[va])
    return final


def paired_bootstrap_delta(y, pred_base, pred_tree, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    deltas = []
    for _ in range(n):
        s = rng.choice(idx, size=len(idx), replace=True)
        deltas.append(r2(y[s], pred_tree[s]) - r2(y[s], pred_base[s]))
    return float(np.mean(deltas)), float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def tree_factory():
    return LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=15,
        min_child_samples=50, subsample=0.8, colsample_bytree=0.8,
        random_state=0, verbosity=-1,
    )


def run_p4_p6(df: pd.DataFrame) -> pd.DataFrame:
    frame, _ = build_frame(df)
    full_cols = list(frame.columns)
    X_full = frame[full_cols].astype(float).values  # LightGBM 原生处理 NaN
    rows: list[dict] = []

    fold_series = {
        seed: pd.read_csv(FOLDS_PATH)[f"fold_seed{seed}"].values for seed in SEEDS
    }

    # ---------- P6：三任务最优基线 × 3 种子 ----------
    y1 = df[T1_TARGET].astype(float).values
    y3 = df[T3_TARGET].astype(float).values
    X1 = frame[["weekly_travel_distance_km"]].astype(float).values
    X3 = frame[["daily_commute_km"]].astype(float).values

    for task, X, y, base_fp, p in [
        ("T1", X1, y1, ratio_base(), 1),
        ("T3", X3, y3, origin_ols_base(), 1),
    ]:
        r2s, adj, rms, maes = [], [], [], []
        for seed in SEEDS:
            fold = fold_series[seed]
            pred = np.full(len(y), np.nan)
            for k in np.unique(fold):
                tr = fold != k; va = fold == k
                pred[va] = base_fp(X[tr], y[tr])(X[va])
            m = pooled_metrics(y, pred, p)
            r2s.append(m["r2"]); adj.append(m["adj_r2"]); rms.append(m["rmse"]); maes.append(m["mae"])
        rows.append({
            "task": task, "stage": "P6", "model": "final_baseline",
            "r2_mean": round(float(np.mean(r2s)), 6),
            "r2_std": round(float(np.std(r2s)), 6),
            "adj_r2_mean": round(float(np.mean(adj)), 6),
            "adj_r2_std": round(float(np.std(adj)), 6),
            "rmse_mean": round(float(np.mean(rms)), 4),
            "mae_mean": round(float(np.mean(maes)), 4),
            "delta_boot_mean": np.nan, "delta_boot_lo": np.nan, "delta_boot_hi": np.nan,
            "notes": f"{task} 最优基线 5折×3种子",
        })

    # T2 级联 × 3 种子
    y2 = df[T2_TARGET].astype(float).values
    r2s, adj, rms, maes = [], [], [], []
    for seed in SEEDS:
        fold = fold_series[seed]
        _, cost_physics = t2_cascade(df, fold, t1_model="ratio")
        m = pooled_metrics(y2, cost_physics, 2)
        r2s.append(m["r2"]); adj.append(m["adj_r2"]); rms.append(m["rmse"]); maes.append(m["mae"])
    rows.append({
        "task": "T2", "stage": "P6", "model": "final_cascade",
        "r2_mean": round(float(np.mean(r2s)), 6), "r2_std": round(float(np.std(r2s)), 6),
        "adj_r2_mean": round(float(np.mean(adj)), 6), "adj_r2_std": round(float(np.std(adj)), 6),
        "rmse_mean": round(float(np.mean(rms)), 4), "mae_mean": round(float(np.mean(maes)), 4),
        "delta_boot_mean": np.nan, "delta_boot_lo": np.nan, "delta_boot_hi": np.nan,
        "notes": "T2 严格级联 cost_physics 5折×3种子",
    })

    # ---------- P4：树模型残差确认（种子 42 单一外层折） ----------
    fold = fold_series[42]
    for task, X, y, base_fp in [
        ("T1", X1, y1, ratio_base()),
        ("T3", X3, y3, origin_ols_base()),
    ]:
        base_pred = np.full(len(y), np.nan)
        for k in np.unique(fold):
            tr = fold != k; va = fold == k
            base_pred[va] = base_fp(X[tr], y[tr])(X[va])
        tree_pred = residual_boost_oof(X, X_full, y, fold, base_fp, tree_factory)
        d_mean, d_lo, d_hi = paired_bootstrap_delta(y, base_pred, tree_pred)
        rows.append({
            "task": task, "stage": "P4", "model": "residual_lgbm",
            "r2_mean": round(r2(y, tree_pred), 6), "r2_std": np.nan,
            "adj_r2_mean": np.nan, "adj_r2_std": np.nan,
            "rmse_mean": np.nan, "mae_mean": np.nan,
            "delta_boot_mean": round(d_mean, 6), "delta_boot_lo": round(d_lo, 6),
            "delta_boot_hi": round(d_hi, 6),
            "notes": f"{task} 树模型相对基线 ΔR²（Bootstrap 95% CI）",
        })

    # T2 残差确认：真实成本 - cost_physics 的树模型
    _, cp = t2_cascade(df, fold, t1_model="ratio")
    resid2 = y2 - cp
    pred2 = np.full(len(y2), np.nan)
    for k in np.unique(fold):
        tr = fold != k; va = fold == k
        t = tree_factory(); t.fit(X_full[tr], resid2[tr])
        pred2[va] = cp[va] + t.predict(X_full[va])
    d_mean, d_lo, d_hi = paired_bootstrap_delta(y2, cp, pred2)
    rows.append({
        "task": "T2", "stage": "P4", "model": "residual_lgbm",
        "r2_mean": round(r2(y2, pred2), 6), "r2_std": np.nan,
        "adj_r2_mean": np.nan, "adj_r2_std": np.nan,
        "rmse_mean": np.nan, "mae_mean": np.nan,
        "delta_boot_mean": round(d_mean, 6), "delta_boot_lo": round(d_lo, 6),
        "delta_boot_hi": round(d_hi, 6),
        "notes": "T2 成本残差树模型相对级联 ΔR²（Bootstrap 95% CI）",
    })

    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "复赛B题" / "B题数据集.csv"))
    args = ap.parse_args()
    df = load_data(args.csv)
    result = run_p4_p6(df)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(result.to_string(index=False))
    print(f"\n已写入 {OUT_PATH}")


if __name__ == "__main__":
    main()
