# 环境隔离配置指南

本文档详细说明如何配置和使用环境隔离，确保项目依赖和配置的隔离性。

## 目录

- [环境隔离方式](#环境隔离方式)
- [虚拟环境配置](#虚拟环境配置)
- [Docker容器配置](#docker容器配置)
- [环境变量配置](#环境变量配置)
- [依赖管理](#依赖管理)
- [最佳实践](#最佳实践)

## 环境隔离方式

项目支持三种环境隔离方式，适用于不同场景：

| 方式 | 适用场景 | 隔离程度 | 设置复杂度 |
|------|---------|---------|-----------|
| Python虚拟环境 | 开发、测试 | 中等 | 低 |
| Docker容器 | 生产部署 | 高 | 中 |
| Conda环境 | 复杂依赖管理 | 高 | 中 |

## 虚拟环境配置

### 快速设置

使用自动化脚本一键设置：

```bash
./setup_venv.sh
```

### 手动设置

#### 1. 创建虚拟环境

```bash
python3 -m venv venv
```

#### 2. 激活虚拟环境

**Linux/macOS:**
```bash
source venv/bin/activate
```

**Windows:**
```bash
venv\Scripts\activate
```

#### 3. 升级pip

```bash
pip install --upgrade pip setuptools wheel
```

#### 4. 安装依赖

**生产依赖（必需）:**
```bash
pip install -r requirements.txt
```

**开发依赖（可选）:**
```bash
pip install -r requirements-dev.txt
```

#### 5. 验证安装

```bash
python check_env.py
```

### 虚拟环境管理

**激活虚拟环境:**
```bash
source venv/bin/activate
```

**停用虚拟环境:**
```bash
deactivate
```

**删除虚拟环境:**
```bash
rm -rf venv
```

## Docker容器配置

### 使用Docker Compose（推荐）

#### 1. 配置环境变量

```bash
cp env.example .env
# 编辑 .env 文件
```

#### 2. 启动服务

```bash
docker-compose up -d
```

#### 3. 查看日志

```bash
docker-compose logs -f
```

#### 4. 停止服务

```bash
docker-compose down
```

### 手动使用Docker

#### 1. 构建镜像

```bash
docker build -t qwen3-omni-api .
```

#### 2. 运行容器

```bash
docker run -d \
  --name qwen3-omni-api \
  --gpus all \
  -p 9999:9999 \
  --env-file .env \
  qwen3-omni-api
```

#### 3. 查看日志

```bash
docker logs -f qwen3-omni-api
```

#### 4. 停止容器

```bash
docker stop qwen3-omni-api
docker rm qwen3-omni-api
```

### Docker镜像说明

- **基础镜像**: `nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04`
- **Python版本**: 3.10
- **非root用户**: 使用 `appuser` 运行应用，提高安全性
- **健康检查**: 自动监控容器健康状态
- **GPU支持**: 支持NVIDIA GPU，通过 `--gpus all` 启用

## 环境变量配置

### 配置文件结构

项目使用 `.env` 文件管理环境变量，`.env` 文件不会被提交到Git。

```
env.example          # 环境变量模板（已提交到Git）
.env                 # 实际环境变量（不提交，本地配置）
.env.local          # 本地覆盖配置（不提交，用于本地开发）
```

### 设置环境变量

#### 1. 复制模板文件

```bash
cp env.example .env
```

#### 2. 编辑配置

使用文本编辑器打开 `.env` 文件，根据实际情况修改：

```bash
# 服务器配置
HOST=192.168.1.22
PORT=9999

# 模型配置
MODEL_NAME=Qwen/Qwen3-Omni-30B-A3B-Instruct
MODEL_DEVICE=cuda
MODEL_TORCH_DTYPE=bfloat16
```

#### 3. 验证配置

运行环境检查脚本：

```bash
python check_env.py
```

### 环境变量优先级

1. 系统环境变量（最高优先级）
2. `.env.local` 文件
3. `.env` 文件
4. `env.example` 默认值（最低优先级）

### 常用配置说明

| 变量名 | 说明 | 默认值 | 示例 |
|--------|------|--------|------|
| `HOST` | 服务器监听地址 | `192.168.1.22` | `192.168.1.22` |
| `PORT` | 服务器端口 | `9999` | `9999` |
| `MODEL_NAME` | 模型名称或路径 | `Qwen/Qwen3-Omni-30B-A3B-Instruct` | `Qwen/Qwen3-Omni-30B-A3B-Instruct` |
| `MODEL_DEVICE` | 运行设备 | 自动检测 | `cuda`, `cpu` |
| `MODEL_TORCH_DTYPE` | PyTorch数据类型 | 自动选择 | `float32`, `float16`, `bfloat16` |
| `CUDA_VISIBLE_DEVICES` | 可见GPU设备 | `0` | `0,1,2,3` |

## 依赖管理

### 依赖文件说明

- **`requirements.txt`**: 生产环境必需依赖
- **`requirements-dev.txt`**: 开发环境额外依赖（包含 `requirements.txt`）

### 安装依赖

#### 生产环境

```bash
pip install -r requirements.txt
```

#### 开发环境

```bash
pip install -r requirements-dev.txt
```

### 更新依赖

#### 生成依赖列表

```bash
pip freeze > requirements-current.txt
```

#### 添加新依赖

1. 添加到 `requirements.txt` 或 `requirements-dev.txt`
2. 运行 `pip install <package>`
3. 更新 `requirements.txt`（如需要固定版本）

### 依赖隔离最佳实践

1. **使用虚拟环境**: 始终在虚拟环境中安装依赖
2. **分离生产和开发依赖**: 生产环境只安装 `requirements.txt`
3. **固定版本范围**: 使用 `>=` 而非 `==`，保留灵活性
4. **定期更新**: 定期检查并更新依赖版本

## 最佳实践

### 开发环境

1. 使用虚拟环境隔离依赖
2. 安装开发依赖用于代码质量检查
3. 使用 `.env` 文件管理配置
4. 定期运行 `check_env.py` 检查环境

```bash
# 1. 设置环境
./setup_venv.sh

# 2. 激活环境
source venv/bin/activate

# 3. 安装开发依赖
pip install -r requirements-dev.txt

# 4. 配置环境变量
cp env.example .env

# 5. 检查环境
python check_env.py
```

### 生产环境

1. 使用Docker容器完全隔离
2. 只安装生产依赖
3. 使用环境变量或配置管理工具
4. 设置适当的资源限制

```bash
# 1. 配置环境变量
cp env.example .env

# 2. 使用Docker Compose
docker-compose up -d

# 3. 监控日志
docker-compose logs -f
```

### 多环境配置

对于多环境（开发、测试、生产），可以使用不同的环境变量文件：

```bash
# 开发环境
.env.development

# 测试环境
.env.test

# 生产环境
.env.production
```

然后在启动时指定：

```bash
# 使用特定环境变量文件
export $(cat .env.production | xargs)
python app.py
```

或在Docker Compose中使用不同的compose文件：

```bash
docker-compose -f docker-compose.prod.yml up -d
```

## 故障排查

### 虚拟环境问题

**问题**: `ModuleNotFoundError`

**解决**:
1. 确认虚拟环境已激活：`echo $VIRTUAL_ENV`
2. 重新安装依赖：`pip install -r requirements.txt`
3. 检查Python路径：`which python`

### Docker问题

**问题**: GPU不可用

**解决**:
1. 确认安装了NVIDIA Docker运行时
2. 检查 `docker-compose.yml` 中的GPU配置
3. 验证NVIDIA驱动：`nvidia-smi`

**问题**: 容器无法启动

**解决**:
1. 查看日志：`docker-compose logs`
2. 检查端口是否被占用
3. 验证 `.env` 文件格式正确

### 环境变量问题

**问题**: 配置未生效

**解决**:
1. 确认 `.env` 文件存在且格式正确
2. 检查环境变量优先级
3. 重启应用使新配置生效

## 安全注意事项

1. **不要提交敏感信息**: `.env` 文件已添加到 `.gitignore`
2. **使用环境变量**: 不要在代码中硬编码敏感配置
3. **限制文件权限**: `.env` 文件应设置为 `chmod 600 .env`
4. **定期轮换密钥**: 定期更新API密钥和密码

## 参考资料

- [Python虚拟环境文档](https://docs.python.org/3/tutorial/venv.html)
- [Docker文档](https://docs.docker.com/)
- [12-Factor App配置](https://12factor.net/config)
