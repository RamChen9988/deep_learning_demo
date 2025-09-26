#!/bin/bash

echo "========================================"
echo "深度学习演示环境安装脚本"
echo "========================================"
echo

# 检查Python环境
echo "检查Python环境..."
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python3，请先安装Python 3.7+"
    echo "下载地址: https://www.python.org/downloads/"
    exit 1
fi

echo "Python环境检查通过"
echo

# 升级pip
echo "升级pip..."
python3 -m pip install --upgrade pip

echo
echo "安装依赖包..."
pip3 install -r requirements.txt

if [ $? -ne 0 ]; then
    echo
    echo "警告: 部分包安装失败，尝试使用国内镜像源..."
    pip3 install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
fi

echo
echo "验证安装..."
python3 check_environment.py

echo
echo "========================================"
echo "安装完成！"
echo "========================================"
echo
echo "运行演示程序:"
echo "  基本演示: python3 examples/01_mnist_basic.py"
echo "  实时演示: python3 examples/03_real_time_demo.py"
echo
