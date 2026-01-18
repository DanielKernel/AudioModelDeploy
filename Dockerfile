# Qwen3 Omni API Docker镜像
# 提供完全隔离的运行环境

FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 创建非root用户
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app

# 复制依赖文件
COPY requirements.txt /app/

# 安装Python依赖
RUN pip3 install --upgrade pip setuptools wheel && \
    pip3 install -r requirements.txt

# 复制应用代码
COPY --chown=appuser:appuser . /app/

# 切换到非root用户
USER appuser

# 暴露端口
EXPOSE 8000

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:9999/health || exit 1

# 启动命令
CMD ["python3", "app.py"]
