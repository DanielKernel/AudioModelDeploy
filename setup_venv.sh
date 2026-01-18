#!/bin/bash
# 虚拟环境设置脚本
# 创建隔离的Python虚拟环境并安装依赖

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Qwen3 Omni 环境隔离设置${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# 检查Python版本
echo -e "${YELLOW}检查Python版本...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python版本: $python_version"

# 检查是否为Python 3.8+
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 8 ]); then
    echo -e "${RED}错误: 需要Python 3.8或更高版本${NC}"
    exit 1
fi

# 虚拟环境目录
VENV_DIR="venv"

# 如果虚拟环境已存在，询问是否删除
if [ -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}虚拟环境已存在: $VENV_DIR${NC}"
    read -p "是否删除并重新创建? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "删除现有虚拟环境..."
        rm -rf "$VENV_DIR"
    else
        echo "使用现有虚拟环境"
    fi
fi

# 创建虚拟环境
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}创建Python虚拟环境...${NC}"
    python3 -m venv "$VENV_DIR"
    echo -e "${GREEN}虚拟环境创建成功${NC}"
fi

# 激活虚拟环境
echo -e "${YELLOW}激活虚拟环境...${NC}"
source "$VENV_DIR/bin/activate"

# 升级pip
echo -e "${YELLOW}升级pip...${NC}"
pip install --upgrade pip setuptools wheel

# 安装基础依赖
echo -e "${YELLOW}安装项目依赖...${NC}"
pip install -r requirements.txt

# 检查是否安装开发依赖
read -p "是否安装开发依赖? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]] && [ -f "requirements-dev.txt" ]; then
    echo -e "${YELLOW}安装开发依赖...${NC}"
    pip install -r requirements-dev.txt
fi

# 验证安装
echo -e "${YELLOW}验证安装...${NC}"
python -c "import torch; import transformers; import fastapi" && {
    echo -e "${GREEN}✓ 核心依赖安装成功${NC}"
} || {
    echo -e "${RED}✗ 核心依赖安装失败${NC}"
    exit 1
}

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}环境设置完成！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "要激活虚拟环境，请运行:"
echo -e "  ${YELLOW}source venv/bin/activate${NC}"
echo ""
echo "要停用虚拟环境，请运行:"
echo -e "  ${YELLOW}deactivate${NC}"
echo ""
