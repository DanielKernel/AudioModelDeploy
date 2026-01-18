# Qwen3 Omni 模型部署 API

在DGX Spark上运行的Qwen3 Omni模型推理API服务器，支持流式交互和多模态输入。

## 功能特性

- ✅ 基于FastAPI的高性能API服务器
- ✅ 支持流式输出（Server-Sent Events, SSE）
- ✅ 支持同步和异步推理
- ✅ 多模态输入支持（文本、图像、音频）
- ✅ 自动GPU/CPU设备检测
- ✅ 完整的API文档（Swagger UI）
- ✅ CORS支持

## 环境要求

- Python 3.8+
- CUDA 11.8+（GPU推理）
- PyTorch 2.0+
- Transformers 4.35+

## 环境隔离说明

本项目支持多种环境隔离方式，确保依赖和环境配置的隔离性：

### 1. Python虚拟环境（推荐开发使用）

- **脚本**: `setup_venv.sh` - 自动创建和配置虚拟环境
- **优点**: 轻量级，快速启动，适合开发和测试
- **用法**: 运行 `./setup_venv.sh` 后使用 `source venv/bin/activate` 激活

### 2. Docker容器（推荐生产使用）

- **文件**: `Dockerfile`, `docker-compose.yml`
- **优点**: 完全隔离，包含所有系统依赖，环境一致性高
- **用法**: `docker-compose up -d`

### 3. 环境变量隔离

- **文件**: `env.example` - 环境变量模板
- **优点**: 配置与代码分离，不同环境使用不同配置
- **用法**: 复制 `env.example` 为 `.env` 并修改（`.env` 不会提交到Git）

### 4. 依赖隔离

- **生产依赖**: `requirements.txt` - 运行必需的最小依赖
- **开发依赖**: `requirements-dev.txt` - 开发工具（测试、格式化等）
- **优点**: 生产环境只安装必要依赖，减少体积和潜在冲突

### 环境检查

运行 `python check_env.py` 可以自动检查：
- Python版本和虚拟环境
- 所有依赖是否正确安装
- CUDA/GPU可用性
- 配置文件是否存在
- 端口是否可用

> 📖 **详细文档**: 查看 [ENVIRONMENT.md](ENVIRONMENT.md) 了解更多环境隔离配置细节

## 快速开始

### 方式一：使用虚拟环境（推荐）

#### 1. 设置虚拟环境

使用自动化脚本创建隔离的Python虚拟环境：

```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

脚本会自动：
- 检查Python版本（需要3.8+）
- 创建虚拟环境
- 安装所有依赖
- 验证安装

#### 2. 激活虚拟环境

```bash
source venv/bin/activate
```

#### 3. 配置环境变量

复制环境变量示例文件并修改：

```bash
cp env.example .env
# 编辑 .env 文件，根据实际情况修改配置
```

主要配置项：
- `MODEL_NAME`: 模型名称或路径
- `HOST`: 服务器监听地址（默认: 0.0.0.0）
- `PORT`: 服务器端口（默认: 8000）
- `MODEL_DEVICE`: 设备（cuda/cpu，默认自动检测）

#### 4. 检查环境

运行环境检查脚本确保配置正确：

```bash
python check_env.py
```

#### 5. 启动服务器

```bash
./start.sh
```

或直接使用Python：

```bash
python app.py
```

### 方式二：使用Docker（完全隔离）

#### 1. 配置环境变量

```bash
cp env.example .env
# 编辑 .env 文件
```

#### 2. 构建并启动

```bash
docker-compose up -d
```

或手动构建：

```bash
docker build -t qwen3-omni-api .
docker run -d --gpus all -p 8000:8000 --env-file .env qwen3-omni-api
```

### 方式三：直接安装（不推荐）

如果不使用虚拟环境（不推荐，可能污染系统环境）：

```bash
pip install -r requirements.txt
cp env.example .env
python app.py
```

### 4. 访问API

- API文档: http://localhost:8000/docs
- 健康检查: http://localhost:8000/health
- 流式聊天: http://localhost:8000/v1/chat/completions

## API使用示例

### 1. 流式聊天（推荐）

使用curl进行流式请求：

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "你好，请介绍一下自己"}
    ],
    "stream": true,
    "temperature": 0.7
  }'
```

Python客户端示例：

```python
import requests
import json

url = "http://localhost:8000/v1/chat/completions"

payload = {
    "messages": [
        {"role": "user", "content": "你好，请介绍一下自己"}
    ],
    "stream": True,
    "temperature": 0.7
}

response = requests.post(url, json=payload, stream=True)

for line in response.iter_lines():
    if line:
        line_text = line.decode('utf-8')
        if line_text.startswith('data: '):
            data = line_text[6:]  # 移除 'data: ' 前缀
            if data == '[DONE]':
                break
            try:
                chunk = json.loads(data)
                if 'choices' in chunk and chunk['choices']:
                    content = chunk['choices'][0].get('delta', {}).get('content', '')
                    if content:
                        print(content, end='', flush=True)
            except json.JSONDecodeError:
                pass
print()  # 换行
```

### 2. 同步聊天

```bash
curl -X POST http://localhost:8000/v1/chat/completions/sync \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "你好"}
    ],
    "temperature": 0.7
  }'
```

### 3. 使用Python requests库

```python
import requests

url = "http://localhost:8000/v1/chat/completions/sync"

response = requests.post(url, json={
    "messages": [
        {"role": "user", "content": "你好，请介绍一下自己"}
    ],
    "temperature": 0.7
})

result = response.json()
print(result['choices'][0]['message']['content'])
```

## API端点

### POST `/v1/chat/completions`

流式聊天完成端点

**请求体:**
```json
{
  "messages": [
    {"role": "user", "content": "你好"}
  ],
  "stream": true,
  "temperature": 0.7,
  "top_p": 0.9,
  "max_tokens": 2048
}
```

**响应格式 (SSE):**
```
data: {"id": "...", "choices": [{"delta": {"content": "你好"}}]}
data: {"id": "...", "choices": [{"delta": {"content": "！"}}]}
data: [DONE]
```

### POST `/v1/chat/completions/sync`

同步聊天完成端点

**请求体:** 同上（`stream`参数可选）

**响应格式:**
```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": "完整的响应内容"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}
```

### GET `/health`

健康检查端点

**响应:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 在DGX Spark上部署

### 1. 环境准备

确保DGX节点上已安装：
- NVIDIA驱动
- CUDA工具包
- Python 3.8+

### 2. 配置GPU

设置可见的GPU设备：

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3  # 使用前4个GPU
export NUM_GPUS=4
```

### 3. 启动服务

```bash
./start.sh
```

或使用nohup后台运行：

```bash
nohup ./start.sh > server.log 2>&1 &
```

### 4. 使用Spark任务提交

可以通过Spark任务管理器启动API服务，确保资源分配正确。

## 性能优化

1. **使用bfloat16精度**: 在支持Tensor Core的GPU上使用bfloat16可提升性能
2. **多GPU部署**: 使用`device_map="auto"`自动分配模型到多个GPU
3. **批处理**: 对于同步API，可以考虑实现批处理功能
4. **模型量化**: 可以使用量化技术减少显存占用

## 故障排查

### 模型加载失败

- 检查模型路径是否正确
- 确认有足够的显存/内存
- 检查transformers版本兼容性

### CUDA内存不足

- 减少`max_tokens`参数
- 使用较小的模型
- 启用模型量化

### 流式输出不工作

- 检查客户端是否正确处理SSE格式
- 确认网络连接稳定
- 查看服务器日志

## 许可证

本项目遵循原Qwen模型的许可证。

## 贡献

欢迎提交Issue和Pull Request！
