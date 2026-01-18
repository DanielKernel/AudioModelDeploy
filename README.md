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

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

复制 `.env.example` 为 `.env` 并根据需要修改：

```bash
cp .env.example .env
```

主要配置项：
- `MODEL_NAME`: 模型名称或路径
- `HOST`: 服务器监听地址（默认: 0.0.0.0）
- `PORT`: 服务器端口（默认: 8000）
- `MODEL_DEVICE`: 设备（cuda/cpu，默认自动检测）

### 3. 启动服务器

使用启动脚本：

```bash
chmod +x start.sh
./start.sh
```

或直接使用Python：

```bash
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
