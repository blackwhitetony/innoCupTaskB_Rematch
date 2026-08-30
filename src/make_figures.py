"""生成论文 EDA 配图到 paper/figures/。

用法：uv run python -m src.make_figures
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .preprocessing import load_data

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "paper" / "figures"
DATA = ROOT / "复赛B题" / "B题数据集.csv"


def style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.25, linewidth=0.5)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(DATA)
    weekly = df["weekly_travel_distance_km"].astype(float)
    daily = df["daily_commute_km"].astype(float)
    energy = df["monthly_energy_consumption_kwh"].astype(float)
    cost = df["monthly_charging_cost"].astype(float)
    fuel = df["fuel_expense_per_month"].astype(float)
    price = df["electricity_cost_per_kwh"].astype(float)

    # 1) 相关热力图
    cols = ["weekly_travel_distance_km", "daily_commute_km", "monthly_energy_consumption_kwh",
            "monthly_charging_cost", "fuel_expense_per_month", "electricity_cost_per_kwh",
            "charging_station_accessibility", "nearest_charging_station_km"]
    labels = ["weekly_km", "daily_km", "energy_kwh", "charging_cost", "fuel_expense", "price_kwh",
              "charge_access", "nearest_station"]
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    ax.set_title("Correlation of core numeric features")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    # 2) T1：周里程 vs 能耗 散点 + 拟合线
    r1 = float(np.mean(energy / weekly))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(weekly, energy, s=4, alpha=0.25, color="steelblue")
    xs = np.linspace(weekly.min(), weekly.max(), 100)
    ax.plot(xs, r1 * xs, color="crimson", lw=2, label=f"ratio fit: r={r1:.3f}")
    ax.set_xlabel("weekly_travel_distance_km"); ax.set_ylabel("monthly_energy_consumption_kwh")
    ax.set_title("Task 1: energy vs weekly distance")
    ax.legend(); style_axes(ax)
    fig.tight_layout(); fig.savefig(FIG_DIR / "task1_energy_scatter.png", dpi=150); plt.close(fig)

    # 3) T1：能效比率分布（揭示生成机制）
    ratio = energy / weekly
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(ratio, bins=60, color="steelblue", alpha=0.8)
    ax.set_xlabel("energy / weekly_distance (kWh per km)")
    ax.set_ylabel("count")
    ax.set_title("Task 1: energy efficiency ratio distribution")
    style_axes(ax)
    fig.tight_layout(); fig.savefig(FIG_DIR / "task1_ratio_dist.png", dpi=150); plt.close(fig)

    # 4) T2：energy×price vs cost（确定性关系）
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(energy * price, cost, s=4, alpha=0.3, color="seagreen")
    lims = [0, float((energy * price).max())]
    ax.plot(lims, lims, color="crimson", lw=2, label="y = x")
    ax.set_xlabel("monthly_energy_consumption_kwh × electricity_cost_per_kwh")
    ax.set_ylabel("monthly_charging_cost")
    ax.set_title("Task 2: cost vs energy×price")
    ax.legend(); style_axes(ax)
    fig.tight_layout(); fig.savefig(FIG_DIR / "task2_cost_scatter.png", dpi=150); plt.close(fig)

    # 5) T3：日通勤 vs 燃油支出 散点 + 拟合线
    beta = float(np.sum(daily * fuel) / np.sum(daily * daily))
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(daily, fuel, s=4, alpha=0.25, color="darkorange")
    xs = np.linspace(daily.min(), daily.max(), 100)
    ax.plot(xs, beta * xs, color="crimson", lw=2, label=f"through-origin OLS: b={beta:.3f}")
    ax.set_xlabel("daily_commute_km"); ax.set_ylabel("fuel_expense_per_month")
    ax.set_title("Task 3: fuel expense vs daily commute")
    ax.legend(); style_axes(ax)
    fig.tight_layout(); fig.savefig(FIG_DIR / "task3_fuel_scatter.png", dpi=150); plt.close(fig)

    # 6) T3：燃油支出分布（负值标注）
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(fuel, bins=80, color="darkorange", alpha=0.8)
    neg = int((fuel < 0).sum())
    ax.axvline(0, color="crimson", ls="--", lw=1.5)
    ax.set_xlabel("fuel_expense_per_month")
    ax.set_ylabel("count")
    ax.set_title(f"Task 3: fuel expense distribution ({neg} negative samples)")
    style_axes(ax)
    fig.tight_layout(); fig.savefig(FIG_DIR / "task3_target_dist.png", dpi=150); plt.close(fig)

    print("图已生成到", FIG_DIR)


if __name__ == "__main__":
    main()
