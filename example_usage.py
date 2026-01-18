"""
Qwen3 Omni API使用示例
展示如何在Python代码中调用API
"""

import requests
import json
from typing import List, Dict, Any, Iterator, Optional


class Qwen3OmniClient:
    """Qwen3 Omni API客户端封装类"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        初始化客户端
        
        Args:
            base_url: API服务器地址
        """
        self.base_url = base_url.rstrip('/')
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.sync_chat_url = f"{self.base_url}/v1/chat/completions/sync"
        self.health_url = f"{self.base_url}/health"
    
    def health_check(self) -> bool:
        """检查API服务器健康状态"""
        try:
            response = requests.get(self.health_url, timeout=5)
            return response.status_code == 200 and response.json().get("model_loaded", False)
        except Exception:
            return False
    
    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None
    ) -> Iterator[str]:
        """
        流式聊天
        
        Args:
            messages: 消息列表，格式: [{"role": "user", "content": "..."}]
            temperature: 温度参数
            top_p: top_p参数
            max_tokens: 最大生成token数
        
        Yields:
            增量生成的文本内容
        """
        payload = {
            "messages": messages,
            "stream": True,
            "temperature": temperature,
            "top_p": top_p
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        response = requests.post(self.chat_url, json=payload, stream=True, timeout=60)
        response.raise_for_status()
        
        for line in response.iter_lines():
            if line:
                line_text = line.decode('utf-8')
                if line_text.startswith('data: '):
                    data = line_text[6:]
                    if data == '[DONE]':
                        break
                    try:
                        chunk = json.loads(data)
                        if 'choices' in chunk and chunk['choices']:
                            delta = chunk['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                yield content
                    except json.JSONDecodeError:
                        pass
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        同步聊天
        
        Args:
            messages: 消息列表
            temperature: 温度参数
            top_p: top_p参数
            max_tokens: 最大生成token数
        
        Returns:
            完整的响应字典
        """
        payload = {
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        response = requests.post(self.sync_chat_url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()
    
    def simple_chat(self, prompt: str, **kwargs) -> str:
        """
        简单聊天（同步）
        
        Args:
            prompt: 用户输入
            **kwargs: 其他生成参数
        
        Returns:
            助手回复
        """
        messages = [{"role": "user", "content": prompt}]
        response = self.chat(messages, **kwargs)
        return response['choices'][0]['message']['content']


# 使用示例
if __name__ == "__main__":
    # 创建客户端
    client = Qwen3OmniClient("http://localhost:8000")
    
    # 检查服务器状态
    if not client.health_check():
        print("错误: API服务器未就绪")
        exit(1)
    
    print("=" * 60)
    print("示例1: 流式聊天")
    print("=" * 60)
    
    messages = [{"role": "user", "content": "你好，请介绍一下自己"}]
    print("问题: 你好，请介绍一下自己")
    print("回答: ", end="", flush=True)
    
    for chunk in client.chat_stream(messages, temperature=0.7):
        print(chunk, end="", flush=True)
    print("\n")
    
    print("=" * 60)
    print("示例2: 同步聊天")
    print("=" * 60)
    
    response = client.simple_chat("用一句话介绍人工智能", temperature=0.7)
    print(f"问题: 用一句话介绍人工智能")
    print(f"回答: {response}\n")
    
    print("=" * 60)
    print("示例3: 多轮对话")
    print("=" * 60)
    
    messages = [
        {"role": "user", "content": "什么是机器学习？"},
    ]
    
    # 第一轮
    response = client.chat(messages)
    assistant_msg = response['choices'][0]['message']['content']
    print(f"用户: 什么是机器学习？")
    print(f"助手: {assistant_msg}\n")
    
    # 添加助手回复到消息历史
    messages.append({"role": "assistant", "content": assistant_msg})
    
    # 第二轮
    messages.append({"role": "user", "content": "它和深度学习有什么区别？"})
    response = client.chat(messages)
    assistant_msg2 = response['choices'][0]['message']['content']
    print(f"用户: 它和深度学习有什么区别？")
    print(f"助手: {assistant_msg2}\n")
