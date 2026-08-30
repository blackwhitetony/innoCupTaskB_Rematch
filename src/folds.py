"""固定折划分（可复现）。

对应 plan.md 第 6.1 节：5 折 shuffled CV × 3 个种子，三任务共用同一组 fold_id。
折分配写入 artifacts/folds/，后续所有实验读取同一文件，保证相同配置可复现同一 OOF。

用法：
    uv run python -m src.folds [--n-splits 5 --seeds 42,2026,7]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]


def generate_folds(
    n_samples: int,
    n_splits: int = 5,
    seeds: list[int] | None = None,
) -> pd.DataFrame:
    seeds = seeds or [42, 2026, 7]
    folds = pd.DataFrame({"row_id": np.arange(n_samples)})
    for seed in seeds:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        col = f"fold_seed{seed}"
        fold_idx = np.empty(n_samples, dtype=np.int8)
        for k, (_, va) in enumerate(kf.split(np.arange(n_samples))):
            fold_idx[va] = k
        folds[col] = fold_idx
    return folds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=str(ROOT / "复赛B题" / "B题数据集.csv"))
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seeds", default="42,2026,7")
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "folds" / "folds.csv"))
    args = ap.parse_args()

    df = pd.read_csv(args.csv, usecols=[0])  # 只读行数
    seeds = [int(s) for s in args.seeds.split(",")]
    folds = generate_folds(len(df), args.n_splits, seeds)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    folds.to_csv(out, index=False)
    print(f"折划分已写入 {out}，形状 {folds.shape}")


if __name__ == "__main__":
    main()
