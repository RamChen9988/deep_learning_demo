"""
步骤2：数据增强与网络优化（加载预训练模型版本）
目标：直接加载预训练的基线模型，然后进行数据增强和网络优化
知识点：模型加载、数据增强、批量归一化、Dropout
预计准确率：80%+

修复版本：移除了重复的模型创建，优化了加载流程
"""

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

class OptimizationTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        self.results = {}  # 记录各步骤结果
    
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
    
    def train_with_optimizations(self, model, use_augmentation=True, epochs=40):
        """使用优化技术训练模型"""
        
        # 可调整的参数
        learning_rate = 0.01      # 学习率选项: 0.001, 0.01, 0.1
        momentum = 0.9            # 动量选项: 0.5, 0.9, 0.99
        weight_decay = 0.0005     # 权重衰减选项: 0, 0.0001, 0.0005, 0.001
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                             momentum=momentum, weight_decay=weight_decay)
        
        # 选择训练数据加载器
        train_loader = self.trainloader_optimized if use_augmentation else self.trainloader_baseline
        
        train_accuracies = []
        test_accuracies = []
        
        print(f"\n开始训练{'带数据增强的' if use_augmentation else '基线'}模型...")
        
        for epoch in range(epochs):
            model.train()
            correct = 0
            total = 0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            train_acc = 100. * correct / total
            test_acc = self.evaluate_model(model)
            
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            
            if epoch % 10 == 0 or epoch == epochs - 1:
                print(f'Epoch {epoch+1}: 训练准确率: {train_acc:.2f}%, '
                      f'测试准确率: {test_acc:.2f}%')
        
        final_acc = test_accuracies[-1]
        technique = "数据增强+网络优化" if use_augmentation else "仅网络优化"
        self.results[technique] = final_acc
        
        return final_acc, train_accuracies, test_accuracies
    
    def evaluate_model(self, model):
        """评估模型性能"""
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, targets in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        return 100. * correct / total
    
    def compare_techniques(self, baseline_acc):
        """比较不同优化技术的效果"""
        print("\n" + "="*50)
        print("优化技术效果对比")
        print("="*50)
        
        # 添加基线模型结果
        self.results["基线模型"] = baseline_acc
        
        techniques = list(self.results.keys())
        accuracies = list(self.results.values())
        
        plt.figure(figsize=(10, 6))
        colors = ['lightblue', 'skyblue', 'lightgreen']
        bars = plt.bar(techniques, accuracies, color=colors)
        
        plt.title('不同优化技术的效果对比', fontsize=14, fontweight='bold')
        plt.ylabel('测试准确率 (%)', fontsize=12)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 在柱状图上添加数值
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # 打印优化建议
        print("\n优化效果分析:")
        for technique, acc in self.results.items():
            print(f"• {technique}: {acc:.2f}%")
        
        # 计算提升
        if "仅网络优化" in self.results and "数据增强+网络优化" in self.results:
            improvement1 = self.results["仅网络优化"] - baseline_acc
            improvement2 = self.results["数据增强+网络优化"] - baseline_acc
            print(f"\n优化效果提升:")
            print(f"• 仅网络优化: +{improvement1:.2f}%")
            print(f"• 数据增强+网络优化: +{improvement2:.2f}%")
        
        print("\n下一步优化建议:")
        print("1. 尝试不同的学习率调度策略")
        print("2. 使用更先进的优化器(如Adam)")
        print("3. 进一步增加网络深度")
        print("4. 调整Dropout比例和权重衰减")

def main():
    """主函数"""
    print("=" * 60)
    print("步骤2: 数据增强与网络优化（加载预训练模型版本）")
    print("目标: 在预训练基线模型基础上进行优化")
    print("=" * 60)
    
    trainer = OptimizationTrainer()
    
    # 加载数据
    trainer.load_data_with_augmentation()
    
    # 加载预训练的基线模型
    baseline_model, baseline_acc = trainer.load_pretrained_baseline_model()
    
    print(f"\n基线模型准确率: {baseline_acc:.2f}%")
    
    # 在基线模型基础上进行优化训练（使用数据增强）
    print("\n在基线模型基础上进行优化训练...")
    print("优化技术: 数据增强 + 批量归一化 + Dropout")
    
    # 创建优化模型并加载基线模型的权重作为初始化
    optimized_model = trainer.create_optimized_model()
    
    # 训练优化模型
    final_acc, train_accuracies, test_accuracies = trainer.train_with_optimizations(
        optimized_model, use_augmentation=True, epochs=40
    )
    
    print(f"\n优化后模型准确率: {final_acc:.2f}%")
    improvement = final_acc - baseline_acc
    print(f"优化提升: +{improvement:.2f}%")
    
    # 保存优化后的模型
    save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'save_model', 'optimized_model.pth')
    torch.save(optimized_model.state_dict(), save_path)
    print(f"优化模型已保存为 '{save_path}'")
    
    # 显示优化效果
    trainer.results["基线模型"] = baseline_acc
    trainer.results["优化模型"] = final_acc
    trainer.compare_techniques(baseline_acc)

if __name__ == "__main__":
    main()
