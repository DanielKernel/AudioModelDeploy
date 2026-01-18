"""
Qwen3 Omni API测试客户端
演示如何使用流式和非流式API
"""
import requests
import json
import sys


def test_health(base_url: str = "http://localhost:8000"):
    """测试健康检查端点"""
    print("=" * 50)
    print("测试健康检查端点...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"状态码: {response.status_code}")
        print(f"响应: {json.dumps(response.json(), indent=2, ensure_ascii=False)}")
        return response.status_code == 200
    except Exception as e:
        print(f"错误: {str(e)}")
        return False


def test_stream_chat(base_url: str = "http://localhost:8000", prompt: str = "你好，请介绍一下自己"):
    """测试流式聊天"""
    print("=" * 50)
    print("测试流式聊天...")
    print(f"问题: {prompt}")
    print("-" * 50)
    
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "stream": True,
        "temperature": 0.7,
        "max_tokens": 512
    }
    
    try:
        response = requests.post(url, json=payload, stream=True, timeout=60)
        response.raise_for_status()
        
        print("回答: ", end="", flush=True)
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
                            delta = chunk['choices'][0].get('delta', {})
                            content = delta.get('content', '')
                            if content:
                                print(content, end='', flush=True)
                            # 检查是否完成
                            finish_reason = chunk['choices'][0].get('finish_reason')
                            if finish_reason:
                                break
                    except json.JSONDecodeError as e:
                        # 忽略JSON解析错误
                        pass
        print("\n" + "=" * 50)
        return True
    except Exception as e:
        print(f"错误: {str(e)}")
        return False


def test_sync_chat(base_url: str = "http://localhost:8000", prompt: str = "你好"):
    """测试同步聊天"""
    print("=" * 50)
    print("测试同步聊天...")
    print(f"问题: {prompt}")
    print("-" * 50)
    
    url = f"{base_url}/v1/chat/completions/sync"
    payload = {
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 512
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        
        result = response.json()
        if 'choices' in result and result['choices']:
            content = result['choices'][0]['message']['content']
            print(f"回答: {content}")
            
            if 'usage' in result:
                usage = result['usage']
                print(f"\nToken使用情况:")
                print(f"  Prompt tokens: {usage.get('prompt_tokens', 'N/A')}")
                print(f"  Completion tokens: {usage.get('completion_tokens', 'N/A')}")
                print(f"  Total tokens: {usage.get('total_tokens', 'N/A')}")
        
        print("=" * 50)
        return True
    except Exception as e:
        print(f"错误: {str(e)}")
        return False


def main():
    """主函数"""
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    prompt = sys.argv[2] if len(sys.argv) > 2 else "你好，请介绍一下自己"
    
    print(f"API服务器地址: {base_url}")
    print()
    
    # 1. 健康检查
    if not test_health(base_url):
        print("\n健康检查失败，请确保服务器正在运行")
        return
    
    print()
    
    # 2. 流式聊天测试
    test_stream_chat(base_url, prompt)
    
    print()
    
    # 3. 同步聊天测试
    test_sync_chat(base_url, "用一句话介绍一下人工智能")


if __name__ == "__main__":
    main()
