"""
步骤1：基线模型建立
目标：建立简单的CNN模型作为性能基准
知识点：神经网络基础、前向传播、反向传播
预计准确率：70%左右
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class BaselineTraine
    def __init__(self):
        # 设备配置 - 优先使用GPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 记录训练历史
        self.train_losses = []
        self.train_accuracies = []
        self.test_accuracies = []
    
    def load_data(self):
        """加载CIFAR-10数据集"""
        print("正在下载CIFAR-10数据集...")
        
        # 简单的数据预处理
        transform = transforms.Compose([
            transforms.ToTensor(),  # 转换为Tensor，归一化到[0,1]
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化到[-1,1]
        ])
        
        # 下载训练集和测试集
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)
        
        # 创建数据加载器
        self.trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)
        self.testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)
        
        # 类别名称
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                       'dog', 'frog', 'horse', 'ship', 'truck')
        
        print(f"训练集样本数: {len(trainset)}")
        print(f"测试集样本数: {len(testset)}")
        
        return trainset, testset
    
    def create_baseline_model(self):
        """创建基线CNN模型"""
        
        class BaselineCNN(nn.Module):
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
        
        model = BaselineCNN().to(self.device)
        print("基线模型结构:")
        print(model)
        return model
    
    def train_model(self, model, epochs=30):
        """训练模型"""
        print("\n开始训练基线模型...")
        
        # 可调整的参数 - 可以修改这些值观察效果
        learning_rate = 0.01      # 学习率: 0.001(小), 0.01(中), 0.1(大)
        momentum = 0.9            # 动量: 0.5(小), 0.9(中), 0.99(大)
        weight_decay = 0.0001     # 权重衰减: 0(无), 0.0001(小), 0.001(中)
        
        # 损失函数 - 交叉熵损失，适合分类问题
        criterion = nn.CrossEntropyLoss()
        
        # 优化器 - SGD with momentum
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                             momentum=momentum, weight_decay=weight_decay)
        
        for epoch in range(epochs):
            model.train()  # 训练模式
            running_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                # 数据移动到设备
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 梯度清零 - 重要！否则梯度会累积
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 反向传播
                loss.backward()
                
                # 参数更新
                optimizer.step()
                
                # 统计信息
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # 每100个batch打印一次
                if batch_idx % 100 == 99:
                    print(f'Epoch: {epoch+1}, Batch: {batch_idx+1}, '
                          f'Loss: {running_loss/100:.3f}, Acc: {100.*correct/total:.2f}%')
                    running_loss = 0.0
            
            # 每个epoch结束后在测试集上评估
            train_acc = 100. * correct / total
            test_acc = self.evaluate_model(model)
            
            self.train_losses.append(running_loss / len(self.trainloader))
            self.train_accuracies.append(train_acc)
            self.test_accuracies.append(test_acc)
            
            print(f'Epoch {epoch+1} 完成: '
                  f'训练准确率: {train_acc:.2f}%, 测试准确率: {test_acc:.2f}%')
        
        return model
    
    def evaluate_model(self, model):
        """评估模型性能"""
        model.eval()  # 评估模式
        correct = 0
        total = 0
        
        with torch.no_grad():  # 不计算梯度，节省内存
            for inputs, targets in self.testloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        return accuracy
    
    def visualize_results(self):
        """可视化训练结果"""
        plt.figure(figsize=(12, 4))
        
        # 训练损失
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, 'b-', label='训练损失')
        plt.title('训练损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 准确率
        plt.subplot(1, 2, 2)
        plt.plot(self.train_accuracies, 'g-', label='训练准确率')
        plt.plot(self.test_accuracies, 'r-', label='测试准确率')
        plt.title('准确率曲线')
        plt.xlabel('Epoch')
        plt.ylabel('准确率 (%)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        final_test_acc = self.test_accuracies[-1] if self.test_accuracies else 0
        print(f"\n基线模型最终测试准确率: {final_test_acc:.2f}%")
        print("预期结果: 约70%左右")
        print("如果准确率过低，可以尝试:")
        print("1. 增加训练轮数 (epochs)")
        print("2. 调整学习率 (learning_rate)")
        print("3. 增加网络深度")

def main():
    """主函数"""
    print("=" * 60)
    print("步骤1: 基线模型建立")
    print("目标: 理解CNN基础结构，建立性能基准")
    print("=" * 60)
    
    # 创建训练器
    trainer = BaselineTrainer()
    
    # 加载数据
    trainer.load_data()
    
    # 创建模型
    model = trainer.create_baseline_model()
    
    # 训练模型
    model = trainer.train_model(model, epochs=30)
    
    # 评估并可视化结果
    trainer.visualize_results()
    
    # 保存模型
    torch.save(model.state_dict(), 'save_model\\baseline_model.pth')
    print("基线模型已保存为 'save_model\\baseline_model.pth'")

if __name__ == "__main__":
    main()