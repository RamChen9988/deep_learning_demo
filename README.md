# 深度学习演示项目

一个用于教学和演示的深度学习项目，包含多种神经网络模型、优化算法对比和实际业务场景应用。

## 项目特性

- 🧠 **基础神经网络演示**: XOR问题、MNIST手写数字识别
- 📊 **监督学习完整流程**: 回归问题、分类问题的端到端解决方案
- ⚡ **优化算法对比**: SGD、Adam、RMSprop等优化器性能比较
- 🎯 **实际业务应用**: 信用卡欺诈检测、情感分析系统
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
├── models/                          # 神经网络模型
│   ├── simple_nn.py                # 简单全连接网络
│   ├── manual_layers.py            # 手动实现各层
│   ├── optimization_models.py      # 优化算法模型
│   └── __init__.py
├── utils/                           # 工具函数
│   ├── data_loader.py              # 数据加载器
│   ├── visualization.py            # 可视化工具
│   ├── gradient_utils.py           # 梯度工具函数
│   ├── optimization_utils.py       # 优化工具函数
│   └── __init__.py
├── examples/                        # 演示示例
│   ├── D01_mnist_basic.py          # 基础MNIST演示
│   ├── D03_real_time_demo.py       # 实时摄像头演示
│   ├── D04_supervised_learning.py  # 监督学习完整流程
│   ├── D05_optimization_comparison.py # 优化算法对比
│   ├── D06_fraud_detection_demo.py # 欺诈检测案例
│   ├── D07_backpropagation_demo.py # 反向传播手动实现
│   ├── D08_layer_relationships.py  # 神经网络层关系演示
│   ├── D09_sentiment_analysis_demo.py # 情感分析实战案例
│   ├── D10_optimization_advanced.py # 优化算法进阶
│   ├── D10_sentiment_analysis_demo_1.py # 情感分析演示1
│   ├── D10_sentiment_analysis_demo_2.py # 情感分析演示2
│   ├── D10_sentiment_analysis_demo_3.py # 情感分析演示3
│   ├── D10_sentiment_analysis_demo_4.py # 情感分析演示4
│   ├── D10_sentiment_analysis_demo_4_interactive.py # 交互式情感分析
│   ├── D10_sentiment_analysis_improvements_analysis.md # 改进分析文档
│   ├── D10__build_dataset_100K.py  # 构建100K数据集
│   ├── D11_weight_initialization.py # 权重初始化演示
│   ├── D12_gradient_problems.py    # 梯度问题诊断
│   └── D10_test*.py                # 测试文件
├── sentiment_model/                 # 情感分析模型
│   ├── model.h5                    # 训练好的模型
│   ├── tokenizer.pkl               # 分词器
│   └── label_mapping.pkl           # 标签映射
├── sentiment_model_fixed/           # 修复版情感分析模型
├── sentiment_model_pretrained/      # 预训练情感分析模型
├── data/                            # 数据集目录
│   ├── chinese_emotion_analysis_100k.csv # 中文情感分析数据集
│   ├── Simplified_Chinese_Multi-Emotion_Dialogue_Dataset.csv # 中文多情感对话数据集
│   └── mnist/                       # MNIST数据集
├── requirements.txt                 # Python依赖包
├── install.bat                      # Windows安装脚本
├── install.sh                       # Linux/macOS安装脚本
├── check_environment.py             # 环境检查脚本
├── diagnose_model.py                # 模型诊断工具
└── README.md                        # 项目说明
```

## 使用说明

### 基础演示
运行MNIST手写数字识别和XOR问题演示：
```bash
python examples/D01_mnist_basic.py
```

### 实时演示
运行摄像头实时数字识别演示：
```bash
python examples/D03_real_time_demo.py
```

### 监督学习完整流程
运行回归和分类问题的完整演示：
```bash
python examples/D04_supervised_learning.py
```

### 优化算法对比
比较不同优化算法的性能：
```bash
python examples/D05_optimization_comparison.py
```

### 欺诈检测案例
运行信用卡欺诈检测系统演示：
```bash
python examples/D06_fraud_detection_demo.py
```

### 情感分析系统
运行中文情感分析演示：
```bash
python examples/D09_sentiment_analysis_demo.py
```

### 交互式情感分析
运行交互式情感分析演示：
```bash
python examples/D10_sentiment_analysis_demo_4_interactive.py
```

### 优化算法进阶
运行高级优化算法演示：
```bash
python examples/D10_optimization_advanced.py
```

### 权重初始化演示
运行权重初始化方法对比：
```bash
python examples/D11_weight_initialization.py
```

### 梯度问题诊断
运行梯度问题诊断演示：
```bash
python examples/D12_gradient_problems.py
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
- **情感分析系统**: 中文文本情感分类，支持多情感类别
- **不平衡数据处理**: 使用加权采样和损失函数处理类别不平衡
- **业务指标分析**: 精确率、召回率、F1分数、AUC等指标

### 5. 实时演示
- **摄像头数字识别**: 使用摄像头实时识别手写数字
- **交互式界面**: 实时显示识别结果和置信度

### 6. 高级主题
- **反向传播手动实现**: 深入理解神经网络训练原理
- **权重初始化方法**: Xavier、He等初始化方法对比
- **梯度问题诊断**: 梯度消失、梯度爆炸问题及解决方案
- **优化算法进阶**: 动量优化、自适应优化器对比

## 新增内容 - 第五课第一、二学时扩展

### 新增演示文件

#### 1. 优化算法进阶 (`examples/D10_optimization_advanced.py`)
- **动量优化演示**: 对比不同动量参数的效果
- **自适应优化器**: AdaGrad、RMSprop、Adam算法对比
- **超参数敏感性**: 学习率对训练效果的影响分析

#### 2. 权重初始化 (`examples/D11_weight_initialization.py`)
- **初始化方法对比**: Xavier、He、全零等方法的性能比较
- **激活值分布分析**: 展示不同初始化对网络激活的影响
- **问题诊断**: 识别和解决初始化相关问题

#### 3. 梯度问题诊断 (`examples/D12_gradient_problems.py`)
- **梯度消失演示**: 深度网络中的梯度衰减问题
- **梯度爆炸演示**: 大权重导致的训练不稳定
- **解决方案对比**: 梯度裁剪、初始化调整等方法效果

### 教学对应关系

#### PPT幻灯片2 - 动量法
```python
# 对应代码：D10_optimization_advanced.py 中的 demo_momentum_optimization()
debug_momentum_demo()  # 取消注释运行
```

## 依赖包说明

### 核心依赖
- `torch`, `torchvision`: PyTorch深度学习框架
- `numpy`: 数值计算
- `pandas`: 数据处理
- `scikit-learn`: 机器学习工具
- `matplotlib`: 数据可视化
- `seaborn`: 高级数据可视化
- `opencv-python`: 图像处理和摄像头访问
- `tensorflow`: 深度学习框架（用于情感分析）
- `keras`: 高级神经网络API

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
