#!/bin/bash
# Qwen3 Omni模型API服务器启动脚本
# 适用于DGX Spark环境

set -e

# 设置默认环境变量
export HOST=${HOST:-0.0.0.0}
export PORT=${PORT:-8000}
export MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2-VL-7B-Instruct"}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

# 检查CUDA可用性
if command -v nvidia-smi &> /dev/null; then
    echo "检测到NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "警告: 未检测到NVIDIA GPU，将使用CPU模式"
fi

# 激活Python环境（如果有）
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "已激活虚拟环境"
fi

# 检查依赖
echo "检查Python依赖..."
python -c "import torch; import transformers; import fastapi" || {
    echo "错误: 缺少必要的Python依赖"
    echo "请运行: pip install -r requirements.txt"
    exit 1
}

# 启动服务器
echo "启动Qwen3 Omni API服务器..."
echo "服务器地址: http://${HOST}:${PORT}"
echo "API文档: http://${HOST}:${PORT}/docs"
echo ""

exec python app.py
