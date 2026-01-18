#!/usr/bin/env python3
"""
环境检查脚本
检查Python环境、依赖和配置是否正确
"""
import sys
import os
import subprocess
import platform
from typing import List, Tuple, Dict, Optional
import re

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


def get_cuda_version() -> Optional[str]:
    """
    获取系统CUDA版本
    
    Returns:
        CUDA版本字符串（如'11.8'），如果无法检测则返回None
    """
    # 方法1: 从nvidia-smi获取CUDA版本
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version,cuda_version', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # 提取CUDA版本（格式可能是 "12.0" 或 "11.8"）
            lines = result.stdout.strip().split('\n')
            if lines:
                # 尝试从输出中提取CUDA版本
                for line in lines:
                    parts = line.split(',')
                    if len(parts) >= 2:
                        cuda_ver = parts[1].strip()
                        # 提取主版本号（如11.8, 12.0）
                        match = re.search(r'(\d+\.\d+)', cuda_ver)
                        if match:
                            return match.group(1)
    except Exception:
        pass
    
    # 方法2: 从nvcc获取CUDA版本
    try:
        result = subprocess.run(
            ['nvcc', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            # 提取CUDA版本（输出格式：release 11.8, V11.8.xxx）
            match = re.search(r'release (\d+\.\d+)', result.stdout)
            if match:
                return match.group(1)
    except Exception:
        pass
    
    return None


def get_pytorch_cuda_install_command(cuda_version: Optional[str] = None) -> Tuple[str, str, str]:
    """
    根据CUDA版本获取PyTorch安装命令
    
    Args:
        cuda_version: CUDA版本（如'11.8', '12.1'），如果为None则自动检测
    
    Returns:
        (安装命令, PyTorch CUDA版本标识, 检测到的CUDA版本)
    """
    if cuda_version is None:
        cuda_version = get_cuda_version()
    
    # 如果没有检测到CUDA版本，使用默认的CUDA 11.8
    if cuda_version is None:
        cuda_version = "11.8"
    
    # 根据CUDA版本选择PyTorch索引
    # PyTorch官方支持的CUDA版本映射
    cuda_version_major = float(cuda_version) if '.' in cuda_version else float(f"{cuda_version}.0")
    
    if cuda_version_major >= 12.0:
        # CUDA 12.x 使用最新版本
        torch_cuda = "cu121"
        index_url = "https://download.pytorch.org/whl/cu121"
    elif cuda_version_major >= 11.8:
        # CUDA 11.8 使用 cu118
        torch_cuda = "cu118"
        index_url = "https://download.pytorch.org/whl/cu118"
    elif cuda_version_major >= 11.7:
        # CUDA 11.7 使用 cu117
        torch_cuda = "cu117"
        index_url = "https://download.pytorch.org/whl/cu117"
    else:
        # CUDA 11.6及以下，使用 cu116
        torch_cuda = "cu116"
        index_url = "https://download.pytorch.org/whl/cu116"
    
    # 构建安装命令
    install_cmd = (
        f"pip install torch torchvision torchaudio "
        f"--index-url {index_url}"
    )
    
    return install_cmd, torch_cuda, cuda_version


def install_pytorch_cuda(cuda_version: Optional[str] = None, auto_confirm: bool = False) -> Tuple[bool, str]:
    """
    自动安装CUDA版本的PyTorch
    
    Args:
        cuda_version: CUDA版本，如果为None则自动检测
        auto_confirm: 是否自动确认安装（不询问用户）
    
    Returns:
        (成功标志, 消息)
    """
    try:
        install_cmd, torch_cuda, detected_cuda = get_pytorch_cuda_install_command(cuda_version)
        
        print(f"\n{YELLOW}准备安装PyTorch CUDA版本...{NC}")
        print(f"  检测到的CUDA版本: {detected_cuda}")
        print(f"  PyTorch CUDA版本: {torch_cuda}")
        print(f"  安装命令: {install_cmd}")
        
        if not auto_confirm:
            response = input(f"\n{YELLOW}是否继续安装? (y/N): {NC}")
            if response.lower() not in ['y', 'yes']:
                return False, "用户取消安装"
        
        print(f"\n{GREEN}开始安装PyTorch CUDA版本...{NC}")
        
        # 执行安装命令
        result = subprocess.run(
            install_cmd.split(),
            capture_output=False,  # 显示实时输出
            text=True,
            timeout=600  # 10分钟超时
        )
        
        if result.returncode == 0:
            # 验证安装
            try:
                import torch
                if torch.cuda.is_available():
                    return True, f"安装成功！PyTorch CUDA可用，版本: {torch.version.cuda}"
                else:
                    return False, "安装完成但CUDA仍然不可用，可能需要重启Python环境"
            except Exception as e:
                return False, f"安装完成但验证失败: {str(e)}"
        else:
            return False, f"安装失败，退出代码: {result.returncode}"
            
    except subprocess.TimeoutExpired:
        return False, "安装超时（超过10分钟）"
    except Exception as e:
        return False, f"安装过程中出错: {str(e)}"


def check_cuda() -> Tuple[bool, Dict[str, str]]:
    """检查CUDA可用性（支持DGX Spark环境）"""
    info = {}
    
    # 方法1: 检查nvidia-smi（系统级检查，最可靠）
    nvidia_smi_available = False
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and result.stdout.strip():
            nvidia_smi_available = True
            gpu_names = result.stdout.strip().split('\n')
            info['nvidia_smi'] = "可用"
            info['gpu_count_smi'] = str(len(gpu_names))
            if gpu_names:
                info['gpu_0_name'] = gpu_names[0]
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        info['nvidia_smi'] = f"不可用 ({str(e)[:30]})"
    
    # 方法2: 检查PyTorch CUDA支持
    pytorch_cuda_available = False
    try:
        import torch
        pytorch_cuda_available = torch.cuda.is_available()
        
        if pytorch_cuda_available:
            info['pytorch_cuda'] = "可用"
            info['cuda_version'] = torch.version.cuda or "未知"
            info['device_count'] = str(torch.cuda.device_count())
            
            # 尝试获取设备信息
            try:
                if torch.cuda.device_count() > 0:
                    info['device_0'] = torch.cuda.get_device_name(0)
                    # 尝试创建一个tensor来验证CUDA真正可用
                    try:
                        test_tensor = torch.tensor([1.0]).cuda()
                        info['cuda_test'] = "通过"
                        del test_tensor
                    except Exception as e:
                        info['cuda_test'] = f"失败 ({str(e)[:30]})"
            except Exception as e:
                info['device_error'] = str(e)[:50]
        else:
            info['pytorch_cuda'] = "不可用"
    except ImportError:
        info['pytorch'] = "未安装"
    
    # 综合判断：如果nvidia-smi可用，则认为CUDA可用（即使PyTorch暂时检测不到）
    # 这在DGX Spark环境中很重要，因为GPU可能已经配置但PyTorch需要特定版本
    cuda_available = nvidia_smi_available or pytorch_cuda_available
    
    if cuda_available:
        info['available'] = "是"
        # 如果PyTorch CUDA不可用但nvidia-smi可用，给出提示
        if nvidia_smi_available and not pytorch_cuda_available:
            info['warning'] = "检测到GPU但PyTorch CUDA不可用，可能需要安装CUDA版本的PyTorch"
    else:
        info['available'] = "否"
    
    return cuda_available, info


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
    
    if 'pytorch' in cuda_info and cuda_info['pytorch'] == "未安装":
        print_check("CUDA", False, "PyTorch未安装，无法检测CUDA")
    else:
        print_check("CUDA可用", cuda_available, cuda_info.get('available', '未知'))
        
        # 显示nvidia-smi检测结果
        if 'nvidia_smi' in cuda_info:
            smi_status = "✓" if cuda_info['nvidia_smi'] == "可用" else "✗"
            print(f"  {smi_status} nvidia-smi: {cuda_info['nvidia_smi']}")
            if 'gpu_count_smi' in cuda_info:
                print(f"    - GPU数量(nvidia-smi): {cuda_info['gpu_count_smi']}")
            if 'gpu_0_name' in cuda_info:
                print(f"    - GPU 0: {cuda_info['gpu_0_name']}")
        
        # 显示PyTorch CUDA检测结果
        if 'pytorch_cuda' in cuda_info:
            pytorch_status = "✓" if cuda_info['pytorch_cuda'] == "可用" else "✗"
            print(f"  {pytorch_status} PyTorch CUDA: {cuda_info['pytorch_cuda']}")
            if 'cuda_version' in cuda_info:
                print(f"    - CUDA版本: {cuda_info['cuda_version']}")
            if 'device_count' in cuda_info:
                print(f"    - GPU数量(PyTorch): {cuda_info['device_count']}")
            if 'device_0' in cuda_info:
                print(f"    - GPU 0: {cuda_info['device_0']}")
            if 'cuda_test' in cuda_info:
                test_status = "✓" if cuda_info['cuda_test'] == "通过" else "✗"
                print(f"    {test_status} CUDA测试: {cuda_info['cuda_test']}")
        
        # 显示警告并提供建议
        if 'warning' in cuda_info:
            print(f"  ⚠ 警告: {cuda_info['warning']}")
            
            # 如果检测到GPU但PyTorch CUDA不可用，提供安装建议
            if (cuda_info.get('nvidia_smi') == "可用" and 
                cuda_info.get('pytorch_cuda') == "不可用"):
                print(f"\n{YELLOW}建议:{NC}")
                print(f"  检测到GPU但PyTorch CUDA不可用。")
                print(f"  建议重新安装依赖以安装CUDA版本的PyTorch：")
                print(f"    1. 运行: ./install_dependencies.sh")
                print(f"    2. 或重新运行: ./setup_venv.sh")
                print(f"  这将自动检测CUDA版本并安装对应的PyTorch。")
                
                # 如果install_dependencies.sh存在，提供快速安装选项
                if os.path.exists("install_dependencies.sh"):
                    print(f"\n{YELLOW}是否现在运行 install_dependencies.sh 安装CUDA版本PyTorch? (y/N): {NC}", end="")
                    try:
                        response = input()
                        if response.lower() in ['y', 'yes']:
                            print(f"{GREEN}运行安装脚本...{NC}")
                            import subprocess
                            result = subprocess.run(
                                ['bash', 'install_dependencies.sh'],
                                capture_output=False
                            )
                            if result.returncode == 0:
                                print(f"{GREEN}安装完成！请重新运行 check_env.py 验证。{NC}")
                            else:
                                print(f"{RED}安装失败。{NC}")
                    except (KeyboardInterrupt, EOFError):
                        print(f"\n{YELLOW}已取消。{NC}")
    
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
