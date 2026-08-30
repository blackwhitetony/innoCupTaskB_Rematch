#!/usr/bin/env bash
# AutoDL 全流程复现：Schema 校验 -> 固定折 -> P2/P3 基线+消融 -> P4/P6 确认
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

DATA_PATH="${1:-复赛B题/B题数据集.csv}"

echo "==> [1/4] Schema 校验"
uv run python -m src.schema_check --csv "$DATA_PATH"

echo "==> [2/4] 生成固定折划分"
uv run python -m src.folds

echo "==> [3/4] P2 稀疏基线 + P3 特征消融"
uv run python -m src.experiments --stage all

echo "==> [4/4] P4 树模型残差确认 + P6 5折×3种子"
uv run python -m src.finalize

echo "==> 全部完成。结果：artifacts/metrics/runs.csv 与 finalize_runs.csv"
