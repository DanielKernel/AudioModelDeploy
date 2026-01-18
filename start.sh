#!/bin/bash
# Qwen3 Omni模型API服务器启动脚本
# 适用于DGX Spark环境

set -e

# 设置默认环境变量
export HOST=${HOST:-192.168.1.22}
export PORT=${PORT:-9999}
export MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen3-Omni-30B-A3B-Instruct"}
export LOG_LEVEL=${LOG_LEVEL:-INFO}

# 检查CUDA可用性
if command -v nvidia-smi &> /dev/null; then
    echo "检测到NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "警告: 未检测到NVIDIA GPU，将使用CPU模式"
fi

# 检查是否在虚拟环境中
if [ -z "$VIRTUAL_ENV" ]; then
    # 如果不在虚拟环境中，尝试激活项目虚拟环境
    if [ -d "venv" ]; then
        source venv/bin/activate
        echo "已激活项目虚拟环境: venv"
    else
        echo "警告: 未检测到虚拟环境"
        echo "建议: 运行 ./setup_venv.sh 创建虚拟环境"
        read -p "是否继续? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
else
    echo "当前虚拟环境: $VIRTUAL_ENV"
fi

# 检查依赖
echo "检查Python依赖..."
python -c "import torch; import transformers; import fastapi" || {
    echo "错误: 缺少必要的Python依赖"
    echo "请运行以下命令之一:"
    echo "  1. ./setup_venv.sh (推荐)"
    echo "  2. pip install -r requirements.txt"
    exit 1
}

# 检查环境变量文件
if [ ! -f ".env" ]; then
    if [ -f "env.example" ]; then
        echo "警告: .env 文件不存在"
        echo "建议: cp env.example .env 并修改配置"
    fi
fi

# 启动服务器
echo "启动Qwen3 Omni API服务器..."
echo "服务器地址: http://${HOST}:${PORT}"
echo "API文档: http://${HOST}:${PORT}/docs"
echo ""

exec python app.py
