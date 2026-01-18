"""
配置文件
管理应用的各种配置参数
"""
import os
from typing import Optional


class Config:
    """应用配置类"""
    
    # 服务器配置
    HOST: str = os.getenv("HOST", "192.168.1.22")
    PORT: int = int(os.getenv("PORT", "9999"))
    
    # 模型配置
    MODEL_NAME: str = os.getenv(
        "MODEL_NAME",
        "Qwen/Qwen3-Omni-30B-A3B-Instruct"  # 默认使用Qwen3 Omni模型
    )
    MODEL_DEVICE: Optional[str] = os.getenv("MODEL_DEVICE", None)  # None表示自动检测
    MODEL_TORCH_DTYPE: Optional[str] = os.getenv("MODEL_TORCH_DTYPE", None)  # None表示自动选择
    
    # 推理配置
    DEFAULT_TEMPERATURE: float = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
    DEFAULT_TOP_P: float = float(os.getenv("DEFAULT_TOP_P", "0.9"))
    DEFAULT_MAX_TOKENS: Optional[int] = int(os.getenv("DEFAULT_MAX_TOKENS", "65535")) if os.getenv("DEFAULT_MAX_TOKENS") else 65535
    
    # 日志配置
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    
    # CORS配置
    CORS_ORIGINS: list = os.getenv("CORS_ORIGINS", "*").split(",") if os.getenv("CORS_ORIGINS") else ["*"]
    
    # Spark/DGX环境配置
    NUM_GPUS: int = int(os.getenv("NUM_GPUS", "1"))
    CUDA_VISIBLE_DEVICES: Optional[str] = os.getenv("CUDA_VISIBLE_DEVICES", None)
    
    @classmethod
    def get_torch_dtype(cls):
        """获取torch数据类型"""
        import torch
        if cls.MODEL_TORCH_DTYPE is None:
            return None
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(cls.MODEL_TORCH_DTYPE.lower(), None)
