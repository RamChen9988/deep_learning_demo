# 深度学习演示项目

一个用于教学和演示的深度学习项目，包含多种神经网络模型和实时演示功能。

## 项目特性

- 🧠 **基础神经网络演示**: XOR问题、MNIST手写数字识别
- 📊 **可视化工具**: 训练过程可视化、激活函数演示
- 🎥 **实时演示**: 摄像头实时数字识别
- 📚 **教学友好**: 代码注释详细，适合学习

## 环境要求

- Python 3.7+
- 支持CUDA的GPU（可选，可CPU运行）

## 快速安装

### Windows用户
双击运行 `install.bat` 文件，或命令行执行：
```bash
install.bat
```

### Linux/macOS用户
在终端中执行：
```bash
chmod +x install.sh
./install.sh
```

### 手动安装
```bash
# 升级pip
python -m pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt

# 如果安装失败，使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

## 项目结构

```
deep_learning_demo/
├── models/              # 神经网络模型
│   ├── simple_nn.py    # 简单全连接网络
│   └── __init__.py
├── utils/               # 工具函数
│   ├── data_loader.py  # 数据加载器
│   ├── visualization.py # 可视化工具
│   └── __init__.py
├── examples/            # 演示示例
│   ├── 01_mnist_basic.py      # 基础演示
│   └── 03_real_time_demo.py   # 实时演示
├── data/                # 数据集目录
├── requirements.txt     # Python依赖包
├── install.bat         # Windows安装脚本
├── install.sh          # Linux/macOS安装脚本
└── README.md           # 项目说明
```

## 使用说明

### 基础演示
运行MNIST手写数字识别和XOR问题演示：
```bash
python examples/01_mnist_basic.py
```

### 实时演示
运行摄像头实时数字识别演示：
```bash
python examples/03_real_time_demo.py
```

### 功能说明

1. **XOR问题演示**: 展示神经网络如何解决非线性可分问题
2. **MNIST分类**: 手写数字识别，包含训练和测试过程
3. **激活函数演示**: 可视化不同激活函数的效果
4. **实时摄像头识别**: 使用摄像头实时识别手写数字

## 依赖包说明

### 核心依赖
- `torch`, `torchvision`: PyTorch深度学习框架
- `numpy`: 数值计算
- `matplotlib`: 数据可视化
- `opencv-python`: 图像处理和摄像头访问

### 可选依赖
- `jupyter`: 交互式笔记本支持
- `seaborn`: 高级数据可视化
- `pandas`: 数据处理

## 常见问题

### 安装问题
**Q: 安装PyTorch时网络连接超时**
A: 使用国内镜像源：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

**Q: 摄像头无法打开**
A: 检查摄像头权限，或使用虚拟摄像头软件

### 运行问题
**Q: 程序提示缺少模块**
A: 确保已安装所有依赖包，重新运行安装脚本

**Q: MNIST数据集下载失败**
A: 程序会自动重试，或手动下载数据集到 `data/mnist/` 目录

## 开发说明

### 添加新模型
在 `models/` 目录下创建新的Python文件，继承 `nn.Module` 类

### 添加新演示
在 `examples/` 目录下创建新的演示文件

### 代码规范
- 使用有意义的变量名和函数名
- 添加详细的注释说明
- 遵循PEP 8代码风格

## 许可证

本项目仅用于教学和演示目的。

## 联系方式

如有问题或建议，请通过项目仓库提交Issue。
