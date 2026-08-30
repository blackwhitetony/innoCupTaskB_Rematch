#!/usr/bin/env bash
# AutoDL 环境初始化：安装 uv 并同步依赖（首次进入实例时执行一次）
set -euo pipefail

echo "==> 安装 uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# 使 uv 在当前 shell 可用
export PATH="$HOME/.local/bin:$PATH"
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"

echo "==> uv 版本"
uv --version

echo "==> 固定 Python 3.11 并同步依赖"
uv python pin 3.11
uv sync

echo "==> 依赖一致性检查"
uv pip check

echo "==> 完成。接下来：上传数据 CSV 后执行 bash autodl/run.sh"
