#!/bin/bash
set -e

echo "============================================"
echo "   🎭 多模态情感识别系统 - 一键启动"
echo "============================================"
echo

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# ========== 国内镜像源配置 ==========
export PIP_INDEX_URL="https://mirrors.aliyun.com/pypi/simple/"
export HF_ENDPOINT="https://hf-mirror.com"
export MODELSCOPE_CACHE="$SCRIPT_DIR/.cache/modelscope"
export HF_HOME="$SCRIPT_DIR/.cache/huggingface"

# ========== 环境搭建 ==========

# 检查 uv 是否安装
if ! command -v uv &>/dev/null; then
    echo "[0/4] 安装 uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env" 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

# 创建虚拟环境（如果不存在）
if [ ! -d ".venv" ]; then
    echo "[1/4] 创建虚拟环境 (Python 3.11)..."
    uv venv --python 3.11
fi

echo "[1/4] 激活虚拟环境..."
source .venv/bin/activate

echo "[2/4] 安装依赖..."
uv pip install -r requirements.txt

# ========== 预下载模型权重 ==========
echo "[3/4] 检查并下载模型权重（使用国内镜像加速）..."

python - <<'PYEOF'
import os, sys

def download_hf_model(model_id, desc):
    """从 HuggingFace (hf-mirror) 下载模型"""
    try:
        from huggingface_hub import snapshot_download
        cache_dir = os.environ.get("HF_HOME", None)
        print(f"  → 检查 {desc} [{model_id}]...")
        snapshot_download(
            repo_id=model_id,
            cache_dir=os.path.join(cache_dir, "hub") if cache_dir else None,
        )
        print(f"    ✓ {desc} 已就绪")
    except Exception as e:
        print(f"    ✗ {desc} 下载失败: {e}")
        print(f"      模型将在首次使用时重试下载")

def download_ms_model(model_id, desc):
    """从 ModelScope 下载模型"""
    try:
        from modelscope.hub.snapshot_download import snapshot_download
        cache_dir = os.environ.get("MODELSCOPE_CACHE", None)
        print(f"  → 检查 {desc} [{model_id}]...")
        snapshot_download(model_id, cache_dir=cache_dir)
        print(f"    ✓ {desc} 已就绪")
    except Exception as e:
        print(f"    ✗ {desc} 下载失败: {e}")
        print(f"      模型将在首次使用时重试下载")

def download_funasr_model(model_id, desc):
    """通过 FunASR 下载 emotion2vec 模型"""
    try:
        print(f"  → 检查 {desc} [{model_id}]...")
        from funasr import AutoModel
        AutoModel(model=model_id, disable_update=True)
        print(f"    ✓ {desc} 已就绪")
    except Exception as e:
        print(f"    ✗ {desc} 下载失败: {e}")
        print(f"      模型将在首次使用时重试下载")

print()
print("--- 文本模型 (StructBERT 中文情感分类 / ModelScope) ---")
download_ms_model(
    "iic/nlp_structbert_emotion-classification_chinese-base",
    "StructBERT 中文情感分类"
)

print()
print("--- 语音模型 (emotion2vec+ / FunASR) ---")
download_funasr_model(
    "iic/emotion2vec_plus_large",
    "emotion2vec+ large"
)

print()
print("--- 面部模型 (ViT Face Expression / HuggingFace) ---")
download_hf_model(
    "trpakov/vit-face-expression",
    "ViT 面部表情识别"
)

print()
print("模型检查完毕。")
PYEOF

# ========== 启动系统 ==========
echo "[4/4] 启动系统..."
echo
echo "============================================"
echo "   浏览器将自动打开 http://localhost:7860"
echo "   按 Ctrl+C 停止服务"
echo "============================================"
echo

python app.py
