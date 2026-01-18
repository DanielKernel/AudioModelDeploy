#!/usr/bin/env python3
"""
环境检查脚本
检查Python环境、依赖和配置是否正确
"""
import sys
import os
import subprocess
import platform
from typing import List, Tuple, Dict

# 颜色定义
RED = '\033[0;31m'
GREEN = '\033[0;32m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color


def check_python_version() -> Tuple[bool, str]:
    """检查Python版本"""
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        return False, f"Python {version_str} (需要 3.8+)"
    return True, version_str


def check_virtual_env() -> Tuple[bool, str]:
    """检查是否在虚拟环境中"""
    in_venv = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )
    
    if in_venv:
        return True, f"虚拟环境: {sys.prefix}"
    return False, "未检测到虚拟环境 (建议使用虚拟环境)"


def check_dependencies() -> List[Tuple[str, bool, str]]:
    """检查Python依赖"""
    required_packages = [
        ('fastapi', 'FastAPI'),
        ('uvicorn', 'Uvicorn'),
        ('pydantic', 'Pydantic'),
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('accelerate', 'Accelerate'),
        ('numpy', 'NumPy'),
    ]
    
    results = []
    for package, name in required_packages:
        try:
            __import__(package)
            version = __import__(package).__version__
            results.append((name, True, version))
        except ImportError:
            results.append((name, False, "未安装"))
    
    return results


def check_cuda() -> Tuple[bool, Dict[str, str]]:
    """检查CUDA可用性"""
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        info = {}
        
        if cuda_available:
            info['available'] = "是"
            info['version'] = torch.version.cuda or "未知"
            info['device_count'] = str(torch.cuda.device_count())
            if torch.cuda.device_count() > 0:
                info['device_0'] = torch.cuda.get_device_name(0)
        else:
            info['available'] = "否"
        
        return cuda_available, info
    except ImportError:
        return False, {'error': 'PyTorch未安装'}


def check_env_file() -> Tuple[bool, str]:
    """检查环境变量文件"""
    if os.path.exists('.env'):
        return True, ".env文件存在"
    elif os.path.exists('env.example'):
        return False, ".env文件不存在 (可从env.example复制)"
    else:
        return False, "未找到环境配置文件"


def check_port() -> Tuple[bool, str]:
    """检查端口是否可用"""
    import socket
    port = int(os.getenv('PORT', '9999'))
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    result = sock.connect_ex(('localhost', port))
    sock.close()
    
    if result == 0:
        return False, f"端口 {port} 已被占用"
    return True, f"端口 {port} 可用"


def print_header(text: str):
    """打印标题"""
    print(f"\n{BLUE}{'='*50}{NC}")
    print(f"{BLUE}{text}{NC}")
    print(f"{BLUE}{'='*50}{NC}")


def print_check(name: str, success: bool, message: str = ""):
    """打印检查结果"""
    status = f"{GREEN}✓{NC}" if success else f"{RED}✗{NC}"
    status_text = f"{GREEN}通过{NC}" if success else f"{RED}失败{NC}"
    print(f"{status} {name:30} {status_text:10} {message}")


def main():
    """主函数"""
    print_header("Qwen3 Omni 环境检查")
    
    # Python版本
    print("\n[Python环境]")
    success, msg = check_python_version()
    print_check("Python版本", success, msg)
    
    # 虚拟环境
    success, msg = check_virtual_env()
    print_check("虚拟环境", success, msg)
    
    # Python依赖
    print("\n[Python依赖]")
    deps = check_dependencies()
    all_deps_ok = True
    for name, success, version in deps:
        print_check(name, success, version if success else version)
        if not success:
            all_deps_ok = False
    
    # CUDA
    print("\n[GPU/CUDA]")
    cuda_available, cuda_info = check_cuda()
    if 'error' in cuda_info:
        print_check("CUDA", False, cuda_info['error'])
    else:
        print_check("CUDA可用", cuda_available, cuda_info.get('available', '未知'))
        if cuda_available:
            print(f"  - CUDA版本: {cuda_info.get('version', '未知')}")
            print(f"  - GPU数量: {cuda_info.get('device_count', '0')}")
            if 'device_0' in cuda_info:
                print(f"  - GPU 0: {cuda_info['device_0']}")
    
    # 配置文件
    print("\n[配置文件]")
    success, msg = check_env_file()
    print_check("环境变量文件", success, msg)
    
    # 端口
    print("\n[网络]")
    success, msg = check_port()
    print_check("端口检查", success, msg)
    
    # 总结
    print_header("检查总结")
    
    issues = []
    if not all_deps_ok:
        issues.append("缺少必要的Python依赖，请运行: pip install -r requirements.txt")
    if not cuda_available and os.getenv('MODEL_DEVICE') == 'cuda':
        issues.append("CUDA不可用，但配置要求使用GPU，将回退到CPU")
    
    if issues:
        print(f"\n{YELLOW}发现以下问题:${NC}")
        for issue in issues:
            print(f"  • {issue}")
        print(f"\n{YELLOW}建议:{NC}")
        print("  1. 使用虚拟环境: ./setup_venv.sh")
        print("  2. 安装依赖: pip install -r requirements.txt")
        print("  3. 配置环境变量: cp env.example .env")
        return 1
    else:
        print(f"\n{GREEN}所有检查通过！环境配置正确。${NC}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
