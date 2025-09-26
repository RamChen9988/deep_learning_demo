@echo off
echo ========================================
echo 深度学习演示环境安装脚本
echo ========================================
echo.

echo 检查Python环境...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python，请先安装Python 3.7+
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python环境检查通过
echo.

echo 升级pip...
python -m pip install --upgrade pip

echo.
echo 安装依赖包...
pip install -r requirements.txt

if %errorlevel% neq 0 (
    echo.
    echo 警告: 部分包安装失败，尝试使用国内镜像源...
    pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
)

echo.
echo 验证安装...
python check_environment.py

echo.
echo ========================================
echo 安装完成！
echo ========================================
echo.
echo 运行演示程序:
echo   基本演示: python examples/01_mnist_basic.py
echo   实时演示: python examples/03_real_time_demo.py
echo.
pause
