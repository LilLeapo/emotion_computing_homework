#!/bin/bash
set -e

echo "============================================"
echo "   🎭 多模态情感识别系统 - 一键启动"
echo "============================================"
echo

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="emotion"

# 检查 conda
if command -v conda &>/dev/null; then
    # 初始化 conda（兼容未 init 的情况）
    eval "$(conda shell.bash hook 2>/dev/null)"

    # 检查环境是否存在
    if ! conda env list | grep -qw "$ENV_NAME"; then
        echo "[1/3] 首次运行，创建 conda 环境 (Python 3.11)..."
        conda create -n "$ENV_NAME" python=3.11 -y
    fi

    echo "[1/3] 激活 conda 环境 $ENV_NAME..."
    conda activate "$ENV_NAME"
else
    echo "[1/3] 未检测到 conda，使用当前 Python 环境..."
    # 创建 venv（如果不存在）
    if [ ! -d "venv" ]; then
        echo "       创建 venv 虚拟环境..."
        python3 -m venv venv
    fi
    source venv/bin/activate
fi

echo "[2/3] 检查并安装依赖..."
pip install -r requirements.txt -q

echo "[3/3] 启动系统..."
echo
echo "============================================"
echo "   浏览器将自动打开 http://localhost:7860"
echo "   按 Ctrl+C 停止服务"
echo "============================================"
echo

python app.py
