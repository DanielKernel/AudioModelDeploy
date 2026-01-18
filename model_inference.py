"""
Qwen3 Omni模型推理模块
处理模型加载、推理和流式输出
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from typing import List, Dict, Any, Optional, AsyncIterator
import logging
import asyncio
from contextlib import asynccontextmanager
import os
import time

logger = logging.getLogger(__name__)


class Qwen3OmniModel:
    """Qwen3 Omni模型封装类"""
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: Optional[torch.dtype] = None
    ):
        """
        初始化模型
        
        Args:
            model_name: 模型名称或路径
            device: 设备（cuda/cpu），如果为None则自动检测
            torch_dtype: torch数据类型，如果为None则自动选择
        """
        self.model_name = model_name or os.getenv(
            "MODEL_NAME",
            "Qwen/Qwen2-VL-7B-Instruct"  # 根据实际模型名称修改
        )
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # 根据设备自动选择数据类型
        if torch_dtype is None:
            if self.device == "cuda":
                self.torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                self.torch_dtype = torch.float32
        else:
            self.torch_dtype = torch_dtype
        
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._is_loaded = False
        
        logger.info(f"初始化模型: {self.model_name}")
        logger.info(f"设备: {self.device}, 数据类型: {self.torch_dtype}")
    
    def load_model(self):
        """加载模型和tokenizer"""
        if self._is_loaded:
            logger.warning("模型已经加载")
            return
        
        try:
            logger.info("开始加载tokenizer...")
            # 尝试加载processor（多模态模型通常有processor）
            try:
                self.processor = AutoProcessor.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                self.tokenizer = self.processor.tokenizer
            except:
                # 如果没有processor，只加载tokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True
                )
                self.processor = None
            
            logger.info("开始加载模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",  # 自动分配设备
                torch_dtype=self.torch_dtype,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # 如果使用processor，确保模型配置正确
            if self.processor:
                self.model.eval()
            
            self._is_loaded = True
            logger.info("模型加载完成")
            
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise
    
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._is_loaded
    
    async def generate(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成完整响应（非流式）
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            top_p: top_p参数
            max_tokens: 最大生成token数
            **kwargs: 其他生成参数
        
        Returns:
            包含响应内容的字典
        """
        if not self._is_loaded:
            raise RuntimeError("模型未加载")
        
        # 在线程池中执行同步推理
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self._generate_sync,
            messages,
            temperature,
            top_p,
            max_tokens,
            kwargs
        )
        
        return result
    
    def _generate_sync(
        self,
        messages: List[Dict[str, Any]],
        temperature: float,
        top_p: float,
        max_tokens: Optional[int],
        kwargs: Dict
    ) -> Dict[str, Any]:
        """同步生成方法（在executor中运行）"""
        try:
            # 准备输入
            if self.processor:
                # 多模态模型使用processor
                text = self._format_messages(messages)
                inputs = self.processor(
                    text=text,
                    images=None,  # 可以根据需要处理图像
                    return_tensors="pt"
                )
                inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            else:
                # 纯文本模型使用tokenizer
                text = self._format_messages(messages)
                inputs = self.tokenizer(
                    text,
                    return_tensors="pt"
                ).to(self.device)
            
            # 生成参数
            generation_config = {
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": temperature > 0,
                **kwargs
            }
            
            if max_tokens:
                generation_config["max_new_tokens"] = max_tokens
            
            # 生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_config
                )
            
            # 解码输出
            if self.processor:
                response_text = self.processor.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
            else:
                response_text = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
            
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": inputs["input_ids"].shape[1],
                    "completion_tokens": outputs[0].shape[0] - inputs["input_ids"].shape[1],
                    "total_tokens": outputs[0].shape[0]
                }
            }
            
        except Exception as e:
            logger.error(f"生成错误: {str(e)}")
            raise
    
    async def generate_stream(
        self,
        messages: List[Dict[str, Any]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        流式生成响应
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            top_p: top_p参数
            max_tokens: 最大生成token数
            **kwargs: 其他生成参数
        
        Yields:
            包含增量响应内容的字典
        """
        if not self._is_loaded:
            raise RuntimeError("模型未加载")
        
        # 在线程池中执行流式推理
        loop = asyncio.get_event_loop()
        queue = asyncio.Queue()
        
        def _generate_stream_sync():
            """同步流式生成方法（在executor中运行）"""
            try:
                # 准备输入
                if self.processor:
                    text = self._format_messages(messages)
                    inputs = self.processor(
                        text=text,
                        images=None,
                        return_tensors="pt"
                    )
                    inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                             for k, v in inputs.items()}
                else:
                    text = self._format_messages(messages)
                    inputs = self.tokenizer(
                        text,
                        return_tensors="pt"
                    ).to(self.device)
                
                # 生成参数
                generation_config = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "do_sample": temperature > 0,
                    **kwargs
                }
                
                if max_tokens:
                    generation_config["max_new_tokens"] = max_tokens
                
                # 流式生成
                generated_text = ""
                input_length = inputs["input_ids"].shape[1]
                
                with torch.no_grad():
                    # 使用generate的流式方式
                    for output in self._stream_generate(inputs, generation_config):
                        # 解码新生成的token
                        if self.processor:
                            new_text = self.processor.decode(
                                output,
                                skip_special_tokens=True
                            )
                        else:
                            new_text = self.tokenizer.decode(
                                output,
                                skip_special_tokens=True
                            )
                        
                        # 提取增量内容
                        if new_text.startswith(generated_text):
                            delta = new_text[len(generated_text):]
                        else:
                            delta = new_text
                        
                        if delta:
                            generated_text = new_text
                            asyncio.run_coroutine_threadsafe(
                                queue.put({
                                    "id": "chatcmpl-stream",
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": self.model_name,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {"content": delta},
                                        "finish_reason": None
                                    }]
                                }),
                                loop
                            )
                
                # 发送结束标记
                asyncio.run_coroutine_threadsafe(
                    queue.put({
                        "id": "chatcmpl-stream",
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": self.model_name,
                        "choices": [{
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop"
                        }]
                    }),
                    loop
                )
                
                # 标记完成
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)
                
            except Exception as e:
                logger.error(f"流式生成错误: {str(e)}")
                asyncio.run_coroutine_threadsafe(
                    queue.put({
                        "error": str(e),
                        "type": "error"
                    }),
                    loop
                )
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)
        
        # 在后台线程中运行流式生成
        await loop.run_in_executor(None, _generate_stream_sync)
        
        # 从队列中读取并yield结果
        while True:
            item = await queue.get()
            if item is None:
                break
            yield item
    
    def _stream_generate(self, inputs: Dict, generation_config: Dict):
        """内部流式生成方法"""
        # 使用generate的流式方式，逐token生成
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask", None)
        input_length = input_ids.shape[1]
        
        # 初始化生成状态
        past_key_values = None
        current_ids = input_ids
        
        max_new_tokens = generation_config.get("max_new_tokens", 2048)
        
        for step in range(max_new_tokens):
            with torch.no_grad():
                # 前向传播
                outputs = self.model(
                    input_ids=current_ids if past_key_values is None else current_ids[:, -1:],
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            
            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]
            
            # 采样下一个token
            if generation_config.get("do_sample", False):
                # 温度采样
                temperature = generation_config.get("temperature", 1.0)
                logits = logits / temperature
                
                # Top-p采样
                if "top_p" in generation_config:
                    from torch.nn.functional import softmax
                    probs = softmax(logits, dim=-1)
                    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                    cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
                    
                    sorted_indices_to_remove = cumsum_probs > generation_config["top_p"]
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                from torch.distributions import Categorical
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # 贪婪解码
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            
            # 检查结束标记
            if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id:
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
            
            # 更新当前序列
            current_ids = torch.cat([current_ids, next_token], dim=-1)
            if attention_mask is not None:
                attention_mask = torch.cat([attention_mask, torch.ones((attention_mask.shape[0], 1), device=attention_mask.device)], dim=-1)
            
            # 返回到当前位置的完整序列（从输入之后的部分）
            yield current_ids[0, input_length:]
    
    def _format_messages(self, messages: List[Dict[str, Any]]) -> str:
        """格式化消息为模型输入格式"""
        # 根据Qwen模型的对话格式进行格式化
        formatted = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                formatted.append(f"System: {content}")
            elif role == "user":
                formatted.append(f"User: {content}")
            elif role == "assistant":
                formatted.append(f"Assistant: {content}")
        
        # Qwen格式通常是特殊的对话格式
        # 这里使用简单格式，实际应该使用模型的chat template
        if hasattr(self.tokenizer, 'apply_chat_template'):
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            return "\n".join(formatted) + "\nAssistant:"
    
    def cleanup(self):
        """清理模型资源"""
        if self.model:
            del self.model
        if self.tokenizer:
            del self.tokenizer
        if self.processor:
            del self.processor
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self._is_loaded = False
        logger.info("模型资源已清理")
