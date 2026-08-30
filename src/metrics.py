"""回归评估指标。

对应 plan.md 第 6.2 节：R²、adj-R²、RMSE、MAE，并对 pooled OOF 统一计算。
"""
from __future__ import annotations

import numpy as np


def r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot


def adj_r2(y_true: np.ndarray, y_pred: np.ndarray, p: int) -> float:
    n = len(y_true)
    if n - p - 1 <= 0:
        return float("nan")
    return 1.0 - (1.0 - r2(y_true, y_pred)) * (n - 1) / (n - p - 1)


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def pooled_metrics(y_true: np.ndarray, y_pred: np.ndarray, p: int) -> dict:
    r2v = r2(y_true, y_pred)
    return {
        "r2": round(r2v, 6),
        "adj_r2": round(adj_r2(y_true, y_pred, p), 6),
        "rmse": round(rmse(y_true, y_pred), 4),
        "mae": round(mae(y_true, y_pred), 4),
        "p": int(p),
        "n": int(len(y_true)),
    }
