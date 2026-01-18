"""
Qwen3 Omni模型推理API服务器
支持流式交互和多种输入格式（文本、图像、音频）
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import uvicorn
import asyncio
import json
import logging
from contextlib import asynccontextmanager

from model_inference import Qwen3OmniModel

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 全局模型实例
model_instance: Optional[Qwen3OmniModel] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时加载模型，关闭时清理资源"""
    global model_instance
    logger.info("正在加载Qwen3 Omni模型...")
    try:
        model_instance = Qwen3OmniModel()
        model_instance.load_model()
        logger.info("模型加载完成")
    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise
    
    yield
    
    # 清理资源
    logger.info("正在清理模型资源...")
    if model_instance:
        model_instance.cleanup()
    logger.info("资源清理完成")


# 创建FastAPI应用
app = FastAPI(
    title="Qwen3 Omni API",
    description="Qwen3 Omni模型推理API，支持流式交互",
    version="1.0.0",
    lifespan=lifespan
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求模型
class ChatMessage(BaseModel):
    role: str  # "user", "assistant", "system"
    content: Any  # 可以是文本字符串或多媒体内容


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = True
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: Optional[int] = None


# 健康检查端点
@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {
        "status": "healthy",
        "model_loaded": model_instance is not None and model_instance.is_loaded()
    }


@app.get("/")
async def root():
    """根端点，返回API信息"""
    return {
        "message": "Qwen3 Omni API Server",
        "endpoints": {
            "/health": "健康检查",
            "/v1/chat/completions": "聊天完成端点（流式）",
            "/v1/chat/completions/sync": "聊天完成端点（同步）",
            "/docs": "API文档"
        }
    }


async def generate_stream(messages: List[Dict], **kwargs):
    """生成流式响应"""
    try:
        async for chunk in model_instance.generate_stream(messages, **kwargs):
            # 格式化SSE格式
            data = json.dumps(chunk, ensure_ascii=False)
            yield f"data: {data}\n\n"
        
        # 发送结束标记
        yield "data: [DONE]\n\n"
    except Exception as e:
        logger.error(f"流式生成错误: {str(e)}")
        error_chunk = {
            "error": str(e),
            "type": "error"
        }
        data = json.dumps(error_chunk, ensure_ascii=False)
        yield f"data: {data}\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    """
    聊天完成端点 - 流式输出
    
    支持流式交互，返回SSE格式的响应
    """
    if not model_instance or not model_instance.is_loaded():
        raise HTTPException(status_code=503, detail="模型未加载")
    
    # 转换消息格式
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    if request.stream:
        # 流式响应
        return StreamingResponse(
            generate_stream(
                messages,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    else:
        # 非流式响应（向后兼容）
        try:
            result = await model_instance.generate(
                messages,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens
            )
            return JSONResponse(content=result)
        except Exception as e:
            logger.error(f"生成错误: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))


@app.post("/v1/chat/completions/sync")
async def chat_completions_sync(request: ChatRequest):
    """
    聊天完成端点 - 同步输出
    
    返回完整的响应，不进行流式输出
    """
    if not model_instance or not model_instance.is_loaded():
        raise HTTPException(status_code=503, detail="模型未加载")
    
    # 转换消息格式
    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
    
    try:
        result = await model_instance.generate(
            messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens
        )
        return JSONResponse(content=result)
    except Exception as e:
        logger.error(f"生成错误: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import os
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        log_level="info",
        workers=1  # 模型推理建议单worker
    )
