#!/usr/bin/env python3
"""
环境验证脚本
检查所有必需的依赖包是否已正确安装
"""

import sys
import importlib

# 必需的包列表
REQUIRED_PACKAGES = [
    'torch',
    'torchvision', 
    'numpy',
    'matplotlib',
    'cv2',  # opencv-python
    'PIL',  # Pillow
    'tqdm',
    'requests'
]

# 可选的包列表
OPTIONAL_PACKAGES = [
    'pandas',
    'scipy',
    'sklearn',  # scikit-learn
    'seaborn',
    'jupyter',
    'ipywidgets'
]

def check_package(package_name, required=True):
    """检查单个包是否可用"""
    try:
        module = importlib.import_module(package_name)
        version = getattr(module, '__version__', '未知版本')
        print(f"✅ {package_name}: {version}")
        return True
    except ImportError:
        if required:
            print(f"❌ {package_name}: 未安装")
            return False
        else:
            print(f"⚠️  {package_name}: 未安装（可选）")
            return True

def main():
    print("=" * 50)
    print("深度学习演示环境验证")
    print("=" * 50)
    print()
    
    # 检查Python版本
    python_version = sys.version.split()[0]
    print(f"Python版本: {python_version}")
    
    # 检查必需的包
    print("\n检查必需包:")
    required_success = True
    for package in REQUIRED_PACKAGES:
        if not check_package(package, required=True):
            required_success = False
    
    # 检查可选的包
    print("\n检查可选包:")
    for package in OPTIONAL_PACKAGES:
        check_package(package, required=False)
    
    # 检查CUDA支持
    print("\n检查CUDA支持:")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA版本: {torch.version.cuda}")
        else:
            print("⚠️  CUDA不可用，将使用CPU")
    except:
        print("❌ 无法检查CUDA状态")
    
    # 总结
    print("\n" + "=" * 50)
    if required_success:
        print("✅ 环境验证通过！所有必需包已安装。")
        print("\n可以运行演示程序:")
        print("  基本演示: python examples/01_mnist_basic.py")
        print("  实时演示: python examples/03_real_time_demo.py")
    else:
        print("❌ 环境验证失败！请安装缺失的包。")
        print("\n运行安装脚本:")
        print("  Windows: install.bat")
        print("  Linux/macOS: ./install.sh")
        print("  或手动安装: pip install -r requirements.txt")
    print("=" * 50)

if __name__ == "__main__":
    main()
