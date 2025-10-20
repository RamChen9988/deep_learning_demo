## 新增内容 - 第五课第一、二学时扩展

### 新增演示文件

#### 1. 优化算法进阶 (`examples/10_optimization_advanced.py`)
- **动量优化演示**: 对比不同动量参数的效果
- **自适应优化器**: AdaGrad、RMSprop、Adam算法对比
- **超参数敏感性**: 学习率对训练效果的影响分析

#### 2. 权重初始化 (`examples/11_weight_initialization.py`)
- **初始化方法对比**: Xavier、He、全零等方法的性能比较
- **激活值分布分析**: 展示不同初始化对网络激活的影响
- **问题诊断**: 识别和解决初始化相关问题

#### 3. 梯度问题诊断 (`examples/12_gradient_problems.py`)
- **梯度消失演示**: 深度网络中的梯度衰减问题
- **梯度爆炸演示**: 大权重导致的训练不稳定
- **解决方案对比**: 梯度裁剪、初始化调整等方法效果

### 教学对应关系

#### PPT幻灯片2 - 动量法
```python
# 对应代码：10_optimization_advanced.py 中的 demo_momentum_optimization()
debug_momentum_demo()  # 取消注释运行

# 深度学习演示项目

一个用于教学和演示的深度学习项目，包含多种神经网络模型、优化算法对比和实际业务场景应用。

## 项目特性

- 🧠 **基础神经网络演示**: XOR问题、MNIST手写数字识别
- 📊 **监督学习完整流程**: 回归问题、分类问题的端到端解决方案
- ⚡ **优化算法对比**: SGD、Adam、RMSprop等优化器性能比较
- 🎯 **实际业务应用**: 信用卡欺诈检测系统
- 📈 **可视化工具**: 训练过程可视化、决策边界、ROC曲线等
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
│   └── manual_layers.py               # 手动实现各层
├── utils/               # 工具函数
│   ├── data_loader.py  # 数据加载器
│   ├── visualization.py # 可视化工具
│   └── gradient_utils.py              # 梯度工具函数
│   └── __init__.py
├── examples/            # 演示示例
│   ├── 01_mnist_basic.py          # 基础MNIST演示
│   ├── 03_real_time_demo.py       # 实时摄像头演示
│   ├── 04_supervised_learning.py  # 监督学习完整流程
│   ├── 05_optimization_comparison.py # 优化算法对比
│   └── 06_fraud_detection_demo.py # 欺诈检测案例
│   ├── 07_backpropagation_demo.py     # 反向传播手动实现
│   ├── 08_layer_relationships.py      # 神经网络层关系演示
│   └── 09_sentiment_analysis_demo.py  # 情感分析实战案例
├── data/                # 数据集目录
├── requirements.txt     # Python依赖包
├── install.bat         # Windows安装脚本
├── install.sh          # Linux/macOS安装脚本
├── check_environment.py # 环境检查脚本
└── README.md           # 项目说明
```

## 使用说明

### 新增功能

examples/07_backpropagation_demo.py：手动实现反向传播的演示，包括各层的实现和训练一个简单模型。

examples/08_autograd_demo.py：使用PyTorch自动求导实现相同模型，对比手动实现。

models/manual_nn.py：手动实现的神经网络层。

utils/backprop_utils.py：反向传播相关的工具函数，如梯度检查。

同时，我们将使用MNIST数据集，因为我们已经有了数据加载的代码。

注意：为了教学清晰，我们将手动实现一个简单的两层网络（全连接层+ReLU+全连接层+Softmax）并进行训练。

步骤：

1. 实现手动神经网络层（Linear、ReLU、Softmax）及其反向传播。
2. 构建一个两层网络，并实现前向和反向传播。
3. 使用梯度下降训练模型，并在MNIST上进行测试。
4. 使用PyTorch的自动求导实现相同模型，比较结果。

-------
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

### 监督学习完整流程
运行回归和分类问题的完整演示：
```bash
python examples/04_supervised_learning.py
```

### 优化算法对比
比较不同优化算法的性能：
```bash
python examples/05_optimization_comparison.py
```

### 欺诈检测案例
运行信用卡欺诈检测系统演示：
```bash
python examples/06_fraud_detection_demo.py
```

## 功能说明

### 1. 基础神经网络演示
- **XOR问题演示**: 展示神经网络如何解决非线性可分问题
- **MNIST分类**: 手写数字识别，包含训练和测试过程
- **激活函数演示**: 可视化不同激活函数的效果

### 2. 监督学习完整流程
- **回归问题**: 房价预测模拟，包含数据生成、模型训练、评估
- **分类问题**: 客户流失预测模拟，包含决策边界可视化
- **完整流程**: 数据预处理、模型构建、训练、评估、可视化

### 3. 优化算法对比
- **优化器比较**: SGD、Momentum SGD、Adam、RMSprop性能对比
- **学习率影响**: 不同学习率对收敛速度和稳定性的影响
- **过拟合预防**: Dropout、权重衰减等正则化方法演示

### 4. 实际业务应用
- **欺诈检测系统**: 模拟信用卡交易数据，检测异常交易
- **不平衡数据处理**: 使用加权采样和损失函数处理类别不平衡
- **业务指标分析**: 精确率、召回率、F1分数、AUC等指标
- **成本效益分析**: 计算欺诈检测系统的经济效益

### 5. 实时演示
- **摄像头数字识别**: 使用摄像头实时识别手写数字
- **交互式界面**: 实时显示识别结果和置信度

## 依赖包说明

### 核心依赖
- `torch`, `torchvision`: PyTorch深度学习框架
- `numpy`: 数值计算
- `pandas`: 数据处理
- `scikit-learn`: 机器学习工具
- `matplotlib`: 数据可视化
- `seaborn`: 高级数据可视化
- `opencv-python`: 图像处理和摄像头访问

### 可选依赖
- `jupyter`: 交互式笔记本支持
- `ipywidgets`: Jupyter交互式控件
- `tqdm`: 进度条显示

## 常见问题

### 安装问题
**Q: 安装PyTorch时网络连接超时**
A: 使用国内镜像源：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

**Q: 摄像头无法打开**
A: 检查摄像头权限，或使用虚拟摄像头软件

**Q: 程序提示缺少模块**
A: 确保已安装所有依赖包，重新运行安装脚本

**Q: MNIST数据集下载失败**
A: 程序会自动重试，或手动下载数据集到 `data/mnist/` 目录

### 运行问题
**Q: 内存不足导致程序崩溃**
A: 减少批量大小或使用更小的模型

**Q: 训练过程太慢**
A: 检查是否使用了GPU，或减少训练轮数

**Q: 可视化图表显示异常**
A: 确保matplotlib正确安装，或检查字体设置

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
