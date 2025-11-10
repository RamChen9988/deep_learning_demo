import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


class BaselineCNN(nn.Module):
    """基线CNN模型（与步骤1相同）"""
    def __init__(self):
        super().__init__()
        # 第一个卷积层: 3输入通道(RGB), 32输出通道, 3x3卷积核, 填充1保持尺寸
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        # 第二个卷积层: 32输入通道, 64输出通道
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        # 最大池化层: 2x2窗口, 步长2 (尺寸减半)
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层1: 64*8*8输入特征, 512输出特征
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        # 全连接层2: 512输入特征, 10输出类别
        self.fc2 = nn.Linear(512, 10)
def forward(self, x):
        # 卷积块1: 卷积 → ReLU → 池化
        # 输入: 32x32x3 → 输出: 16x16x32
        x = self.pool(F.relu(self.conv1(x)))
        
        # 卷积块2: 卷积 → ReLU → 池化  
        # 输入: 16x16x32 → 输出: 8x8x64
        x = self.pool(F.relu(self.conv2(x)))
        
        # 展平特征图: 8x8x64 → 4096维向量
        x = x.view(-1, 64 * 8 * 8)
        
        # 全连接层: 4096 → 512 → 10
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
def load_data_with_augmentation(self):
        """加载带数据增强的数据集"""
        print("正在加载数据增强版本的数据集...")
        
        # 基线版本的数据预处理（简单）
        transform_baseline = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # 优化版本的数据预处理（带数据增强）
        transform_optimized = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),      # 随机水平翻转
            transforms.RandomCrop(32, padding=4),        # 随机裁剪
            transforms.ColorJitter(brightness=0.2, contrast=0.2),  # 颜色抖动
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        # 加载数据集
        trainset_baseline = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_baseline)
        trainset_optimized = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_optimized)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_baseline)
        
        # 创建数据加载器
        self.trainloader_baseline = DataLoader(trainset_baseline, batch_size=128, shuffle=True)
        self.trainloader_optimized = DataLoader(trainset_optimized, batch_size=128, shuffle=True)
        self.testloader = DataLoader(testset, batch_size=128, shuffle=False)
        
        print("数据增强说明:")
        print("• RandomHorizontalFlip: 模拟图像镜像，增加数据多样性")
        print("• RandomCrop: 随机裁剪，让模型关注局部特征")  
        print("• ColorJitter: 颜色变化，增强光照不变性")
        
        return trainset_optimized, testset
        
def load_pretrained_baseline_model(self):
        """加载预训练的基线模型"""
        # 使用相对于项目根目录的路径
        model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'save_model', 'baseline_model.pth')
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"预训练模型文件 {model_path} 不存在，请先运行步骤1训练基线模型")
        
        print(f"正在加载预训练的基线模型: {model_path}")
        
        # 使用文件顶部定义的BaselineCNN类创建模型结构并加载权重
        model = BaselineCNN().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # 评估基线模型性能
        baseline_acc = self.evaluate_model(model)
        print(f"预训练基线模型测试准确率: {baseline_acc:.2f}%")
        
        return model, baseline_acc

def create_optimized_model(self):
        """创建优化后的模型（包含BN和Dropout）"""
        
        class OptimizedCNN(nn.Module):
            def __init__(self):
                super().__init__()
                # 卷积块1: 卷积 → 批量归一化 → ReLU → 池化
                self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
                self.bn1 = nn.BatchNorm2d(64)  # 批量归一化
                
                # 卷积块2
                self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
                self.bn2 = nn.BatchNorm2d(128)
                
                # 卷积块3
                self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
                self.bn3 = nn.BatchNorm2d(256)
                
                self.pool = nn.MaxPool2d(2, 2)
                self.dropout = nn.Dropout(0.5)  # Dropout正则化
                
                # 全连接层
                self.fc1 = nn.Linear(256 * 4 * 4, 512)
                self.fc2 = nn.Linear(512, 256)
                self.fc3 = nn.Linear(256, 10)
                
            def forward(self, x):
                # 卷积块1: 32x32 → 16x16
                x = self.pool(F.relu(self.bn1(self.conv1(x))))
                
                # 卷积块2: 16x16 → 8x8  
                x = self.pool(F.relu(self.bn2(self.conv2(x))))
                
                # 卷积块3: 8x8 → 4x4
                x = self.pool(F.relu(self.bn3(self.conv3(x))))
                
                # 展平
                x = x.view(-1, 256 * 4 * 4)
                
                # 全连接层 + Dropout
                x = F.relu(self.fc1(x))
                x = self.dropout(x)  # 只在训练时起作用
                x = F.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.fc3(x)
                
                return x
        
        model = OptimizedCNN().to(self.device)
        print("\n优化模型改进点:")
        print("• 增加网络深度: 3个卷积层")
        print("• 批量归一化: 加速训练，稳定梯度")
        print("• Dropout: 防止过拟合，提升泛化能力")
        print("• 更多通道数: 增强特征提取能力")
        
        return model