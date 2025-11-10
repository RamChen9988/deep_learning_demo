"""
简化版迁移学习程序
基于预训练模型快速构建图像分类器
只使用最优的微调方法，移除策略比较
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import time
import os
from torchvision.datasets import ImageFolder

# 设置中文字体
plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class SimpleTransferLearning:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
    def setup_data(self, data_dir='../data/'):
        """
        设置数据加载和预处理
        """
        print("设置数据预处理...")
        
        # 数据增强 - 训练集使用更强的增强
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # 预训练模型需要224x224输入
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet统计信息
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 测试集使用简单变换
        test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 加载数据集
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')
        
        trainset = ImageFolder(train_dir, transform=train_transform)
        testset = ImageFolder(test_dir, transform=test_transform)
        
        # 创建数据加载器
        self.trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
        self.testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
        
        print(f"训练样本数: {len(trainset)}")
        print(f"测试样本数: {len(testset)}")
        print(f"类别: {trainset.classes}")
        
        return trainset, testset
    
    def create_model(self, model_name='resnet18'):
        """
        创建迁移学习模型 - 使用最优的微调方法
        """
        print(f"\n创建 {model_name} 迁移学习模型...")
        
        # 加载预训练模型（使用最新的weights API）
        if model_name == 'resnet18':
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif model_name == 'vgg16':
            model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        else:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)  # 默认使用resnet18
        
        # 微调方法：解冻部分层
        if model_name == 'resnet18':
            # 解冻最后2个卷积块
            for name, param in model.named_parameters():
                if 'layer4' in name or 'layer3' in name:  # 解冻后面的层
                    param.requires_grad = True
                else:  # 冻结前面的层
                    param.requires_grad = False
            # 替换分类器
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, len(self.trainloader.dataset.classes))
        elif model_name == 'vgg16':
            # 冻结特征提取部分
            for param in model.features.parameters():
                param.requires_grad = False
            # 替换分类器
            num_features = model.classifier[6].in_features
            model.classifier[6] = nn.Linear(num_features, len(self.trainloader.dataset.classes))
        
        # 移动到设备
        model = model.to(self.device)
        
        # 打印可训练参数数量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"总参数: {total_params:,}, 可训练参数: {trainable_params:,} "
              f"({trainable_params/total_params*100:.1f}%)")
        
        return model
    
    def train_model(self, model, epochs=5):
        """
        训练迁移学习模型
        """
        print("\n开始训练迁移学习模型...")
        
        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=0.001,  # 迁移学习使用较小的学习率
            weight_decay=1e-4
        )
        
        # 学习率调度器
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
        
        train_losses = []
        test_accuracies = []
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            start_time = time.time()
            
            for batch_idx, (inputs, targets) in enumerate(self.trainloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                # 统计信息
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                if batch_idx % 50 == 49:
                    print(f'Batch {batch_idx+1}, Loss: {running_loss/50:.3f}')
                    running_loss = 0.0
            
            # 学习率调度
            scheduler.step()
            
            # 测试准确率
            test_acc = self.evaluate_model(model)
            
            train_losses.append(running_loss / len(self.trainloader))
            test_accuracies.append(test_acc)
            
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{epochs}, 时间: {epoch_time:.1f}s, '
                  f'训练准确率: {100.*correct/total:.2f}%, 测试准确率: {test_acc:.2f}%')
        
        return train_losses, test_accuracies
    
    def evaluate_model(self, model):
        """
        评估模型性能
        """
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
        
        accuracy = 100. * correct / total
        return accuracy
    
    def plot_training_history(self, train_losses, test_accuracies):
        """
        绘制训练历史
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # 绘制损失
        ax1.plot(train_losses, 'b-', linewidth=2)
        ax1.set_title('训练损失', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        # 绘制准确率
        ax2.plot(test_accuracies, 'g-', linewidth=2, marker='o')
        ax2.set_title('测试准确率', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def run_demo(self, data_dir='../data/', model_name='resnet18', epochs=5):
        """
        运行简化的迁移学习演示
        """
        print("=" * 50)
        print("简化版迁移学习演示")
        print("目标: 使用预训练模型快速构建图像分类器")
        print("=" * 50)
        
        # 1. 设置数据
        self.setup_data(data_dir)
        
        # 2. 创建迁移学习模型
        model = self.create_model(model_name)
        
        # 3. 训练模型
        print("\n开始训练...")
        train_losses, test_accuracies = self.train_model(model, epochs)
        
        # 4. 最终评估
        final_accuracy = self.evaluate_model(model)
        print(f"\n🎉 迁移学习模型最终准确率: {final_accuracy:.2f}%")
        
        # 5. 绘制训练历史
        self.plot_training_history(train_losses, test_accuracies)
        
        # 6. 保存模型
        model_path = f'../save_model/simple_transfer_learning_{model_name}.pth'
        torch.save(model.state_dict(), model_path)
        print(f"\n💾 模型已保存为 '{model_path}'")
        
        return model, final_accuracy

def main():
    """主函数"""
    demo = SimpleTransferLearning()
    
    # 运行演示
    model, accuracy = demo.run_demo(
        data_dir='../data/',    # 数据目录
        model_name='resnet18', # 预训练模型
        epochs=5               # 训练轮数
    )
    
    print(f"\n✅ 迁移学习完成!")
    print(f"最终准确率: {accuracy:.2f}%")

if __name__ == "__main__":
    main()
