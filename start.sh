#!/bin/bash
set -e

echo "============================================"
echo "   🎭 多模态情感识别系统 - 一键启动"
echo "============================================"
echo

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# 检查 uv 是否安装
if ! command -v uv &>/dev/null; then
    echo "[0/3] 安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

# 创建虚拟环境（如果不存在）
if [ ! -d ".venv" ]; then
    echo "[1/3] 创建虚拟环境 (Python 3.11)..."
    uv venv --python 3.11
fi

echo "[1/3] 激活虚拟环境..."
source .venv/bin/activate

echo "[2/3] 安装依赖..."
uv pip install -r requirements.txt

echo "[3/3] 启动系统..."
echo
echo "============================================"
echo "   浏览器将自动打开 http://localhost:7860"
echo "   按 Ctrl+C 停止服务"
echo "============================================"
echo

python app.py
