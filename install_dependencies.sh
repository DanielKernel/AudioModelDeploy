#!/bin/bash
# 依赖安装脚本
# 自动检测CUDA版本并安装对应的PyTorch，然后安装其他依赖

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检测CUDA版本
get_cuda_version() {
    # 方法1: 从nvidia-smi获取CUDA版本
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader,nounits 2>/dev/null | head -n 1 | awk '{print $1}')
        if [ ! -z "$CUDA_VERSION" ]; then
            echo "$CUDA_VERSION"
            return 0
        fi
    fi
    
    # 方法2: 从nvcc获取CUDA版本
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/')
        if [ ! -z "$CUDA_VERSION" ]; then
            echo "$CUDA_VERSION"
            return 0
        fi
    fi
    
    return 1
}

# 获取PyTorch安装命令
get_pytorch_install_cmd() {
    local cuda_version=$1
    
    # 如果没有提供CUDA版本，尝试检测
    if [ -z "$cuda_version" ]; then
        cuda_version=$(get_cuda_version)
    fi
    
    # 如果没有检测到CUDA版本，使用默认的CUDA 11.8
    if [ -z "$cuda_version" ]; then
        cuda_version="11.8"
    fi
    
    # 根据CUDA版本选择PyTorch索引
    local cuda_major=$(echo "$cuda_version" | cut -d. -f1)
    local cuda_minor=$(echo "$cuda_version" | cut -d. -f2)
    
    # 使用整数比较，避免bc依赖
    if [ "$cuda_major" -gt 12 ] || ([ "$cuda_major" -eq 12 ] && [ "$cuda_minor" -ge 0 ]); then
        # CUDA 12.x 使用 cu121
        echo "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
    elif [ "$cuda_major" -eq 11 ] && [ "$cuda_minor" -ge 8 ]; then
        # CUDA 11.8+ 使用 cu118
        echo "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
    elif [ "$cuda_major" -eq 11 ] && [ "$cuda_minor" -ge 7 ]; then
        # CUDA 11.7 使用 cu117
        echo "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117"
    else
        # CUDA 11.6及以下，使用 cu116
        echo "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu116"
    fi
}

# 检查GPU可用性
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | grep -q .; then
            return 0
        fi
    fi
    return 1
}

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}安装项目依赖${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查GPU和CUDA
HAS_GPU=false
CUDA_VERSION=""
INSTALL_CUDA_PYTORCH=false

if check_gpu; then
    HAS_GPU=true
    CUDA_VERSION=$(get_cuda_version)
    
    echo -e "${GREEN}✓ 检测到GPU${NC}"
    if [ ! -z "$CUDA_VERSION" ]; then
        echo -e "${GREEN}✓ 检测到CUDA版本: $CUDA_VERSION${NC}"
        INSTALL_CUDA_PYTORCH=true
    else
        echo -e "${YELLOW}⚠ 未检测到CUDA版本，将安装CPU版本的PyTorch${NC}"
    fi
else
    echo -e "${YELLOW}未检测到GPU，将安装CPU版本的PyTorch${NC}"
fi

# 创建临时的requirements文件（排除torch相关依赖）
TEMP_REQUIREMENTS=$(mktemp)
grep -v "^torch" requirements.txt > "$TEMP_REQUIREMENTS" || true

# 安装PyTorch
echo ""
echo -e "${YELLOW}安装PyTorch...${NC}"
if [ "$INSTALL_CUDA_PYTORCH" = true ] && [ ! -z "$CUDA_VERSION" ]; then
    PYTORCH_CMD=$(get_pytorch_install_cmd "$CUDA_VERSION")
    echo -e "${GREEN}使用CUDA版本PyTorch (CUDA $CUDA_VERSION)${NC}"
    echo -e "${YELLOW}执行: $PYTORCH_CMD${NC}"
    eval "$PYTORCH_CMD"
else
    echo -e "${YELLOW}使用CPU版本PyTorch${NC}"
    pip install torch torchvision torchaudio
fi

# 安装其他依赖
echo ""
echo -e "${YELLOW}安装其他依赖...${NC}"
pip install -r "$TEMP_REQUIREMENTS"

# 清理临时文件
rm -f "$TEMP_REQUIREMENTS"

# 验证安装
echo ""
echo -e "${YELLOW}验证安装...${NC}"
if python -c "import torch; import transformers; import fastapi" 2>/dev/null; then
    echo -e "${GREEN}✓ 核心依赖安装成功${NC}"
    
    # 验证PyTorch CUDA是否可用
    if python -c "import torch; print('PyTorch CUDA可用' if torch.cuda.is_available() else 'PyTorch CUDA不可用')" 2>/dev/null | grep -q "可用"; then
        echo -e "${GREEN}✓ PyTorch CUDA可用${NC}"
    else
        if [ "$HAS_GPU" = true ]; then
            echo -e "${YELLOW}⚠ PyTorch CUDA不可用，但已安装CUDA版本${NC}"
        fi
    fi
else
    echo -e "${RED}✗ 核心依赖安装失败${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}依赖安装完成！${NC}"
