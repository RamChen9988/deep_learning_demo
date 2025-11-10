"""
迁移学习实战：站在巨人肩上
使用预训练模型快速构建猫狗分类器

核心思想：
1. 利用在大数据集(ImageNet)上训练好的模型
2. 复用其特征提取能力
3. 只训练最后的分类层，适配我们的任务

优势：
• 训练速度快
• 所需数据少
• 准确率高
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import time
import os
from torchvision.datasets import ImageFolder    

# 设置中文字体
plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class TransferLearningDemo:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
    def setup_data(self):
        """
        设置数据加载和预处理
        注意：这里使用CIFAR-10模拟猫狗分类
        实际项目中应替换为真实的猫狗数据集
        """
        print("设置数据预处理...")
        
        # 数据增强 - 训练集使用更强的增强
        train_transform = transforms.Compose([
            transforms.Resize(224),  # 预训练模型需要224x224输入
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet统计信息
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 测试集使用简单变换
        test_transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 加载CIFAR-10数据集（模拟猫狗分类）
        # 实际项目中应该使用：torchvision.datasets.ImageFolder('path/to/cat_dog_data')
        # 假设数据目录结构：
        # data/
        #   train/
        #     cats/
        #     dogs/
        #   test/
        #     cats/ 
        #     dogs/

        # trainset = ImageFolder('data/train', transform=train_transform)
        # testset = ImageFolder('data/test', transform=test_transform)
       

        trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                              download=True, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                             download=True, transform=test_transform)
        
        # 创建数据加载器
        self.trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)
        self.testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)
        
        print(f"训练样本数: {len(trainset)}")
        print(f"测试样本数: {len(testset)}")
        
        return trainset, testset
    
    def create_transfer_model(self, model_name='resnet18', num_classes=10):
        """
        创建迁移学习模型
        
        参数说明：
        model_name: 预训练模型名称 ('resnet18', 'vgg16', 'alexnet'等)
        num_classes: 我们的任务类别数（猫狗分类是2类）
        """
        print(f"\n创建 {model_name} 迁移学习模型...")
        
        # 方法1：特征提取（固定卷积层，只训练分类器）
        def create_feature_extractor():
            """特征提取方法 - 适合小数据集"""
            # 加载预训练模型
            if model_name == 'resnet18':
                model = models.resnet18(pretrained=True)
            elif model_name == 'vgg16':
                model = models.vgg16(pretrained=True)
            else:
                model = models.alexnet(pretrained=True)
            
            # 冻结所有卷积层参数 - 不更新权重
            for param in model.parameters():
                param.requires_grad = False
            
            # 替换最后的全连接层，适配我们的任务
            if model_name == 'resnet18':
                num_features = model.fc.in_features
                model.fc = nn.Linear(num_features, num_classes)
            elif model_name == 'vgg16':
                num_features = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_features, num_classes)
            else:  # alexnet
                num_features = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_features, num_classes)
            
            print("使用特征提取方法：冻结卷积层，只训练分类器")
            return model
        
        # 方法2：微调（解冻部分层，用较小学习率训练）
        def create_fine_tune_model():
            """微调方法 - 适合中等数据集"""
            # 加载预训练模型
            if model_name == 'resnet18':
                model = models.resnet18(pretrained=True)
            elif model_name == 'vgg16':
                model = models.vgg16(pretrained=True)
            else:
                model = models.alexnet(pretrained=True)
            
            # 只冻结前面的层，解冻后面的层
            if model_name == 'resnet18':
                # 冻结前面的层
                for param in list(model.parameters())[:-20]:  # 只保留最后20层可训练
                    param.requires_grad = False
                # 替换最后的全连接层
                num_features = model.fc.in_features
                model.fc = nn.Linear(num_features, num_classes)
            elif model_name == 'vgg16':
                # 冻结特征提取部分
                for param in model.features.parameters():
                    param.requires_grad = False
                # 替换分类器
                num_features = model.classifier[6].in_features
                model.classifier[6] = nn.Linear(num_features, num_classes)
            
            print("使用微调方法：解冻部分层，用较小学习率训练")
            return model
        
        # 选择迁移学习方法
        # 方法1适合小数据集(<1000样本)，方法2适合中等数据集(1000-10000样本)
        if len(self.trainloader.dataset) < 1000:
            model = create_feature_extractor()
        else:
            model = create_fine_tune_model()
        
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
        
        # 只对需要梯度的参数进行优化
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
                
                if batch_idx % 100 == 99:
                    print(f'Batch {batch_idx+1}, Loss: {running_loss/100:.3f}')
                    running_loss = 0.0
            
            # 学习率调度
            scheduler.step()
            
            # 测试准确率
            test_acc = self.evaluate_model(model)
            
            train_losses.append(running_loss / len(self.trainloader))
            test_accuracies.append(test_acc)
            
            epoch_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{epochs}, 时间: {epoch_time:.1f}s, '
                  f'测试准确率: {test_acc:.2f}%')
        
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
    
    def compare_strategies(self):
        """
        比较不同迁移学习策略的效果
        """
        print("\n比较不同迁移学习策略...")
        
        strategies = {
            '特征提取(冻结所有)': self.create_transfer_model(),
            '微调(解冻部分)': self.create_fine_tune_model()
        }
        
        results = {}
        
        for name, model in strategies.items():
            print(f"\n训练策略: {name}")
            train_losses, test_accuracies = self.train_model(model, epochs=3)
            final_acc = test_accuracies[-1]
            results[name] = final_acc
            print(f"{name} 最终准确率: {final_acc:.2f}%")
        
        # 可视化比较结果
        self.visualize_comparison(results)
        
        return results
    
    def create_fine_tune_model(self):
        """创建微调模型示例"""
        model = models.resnet18(pretrained=True)
        
        # 解冻最后2个卷积块
        for name, param in model.named_parameters():
            if 'layer4' in name or 'layer3' in name:  # 解冻后面的层
                param.requires_grad = True
            else:  # 冻结前面的层
                param.requires_grad = False
        
        # 替换分类器
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 10)
        
        return model.to(self.device)
    
    def visualize_comparison(self, results):
        """
        可视化迁移学习效果对比
        """
        strategies = list(results.keys())
        accuracies = list(results.values())
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(strategies, accuracies, color=['skyblue', 'lightgreen'])
        
        plt.title('迁移学习策略效果对比', fontsize=14, fontweight='bold')
        plt.ylabel('测试准确率 (%)', fontsize=12)
        plt.ylim(0, 100)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.5,
                    f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def run_demo(self):
        """
        运行完整的迁移学习演示
        """
        print("=" * 60)
        print("迁移学习实战演示")
        print("目标: 使用预训练模型快速构建图像分类器")
        print("=" * 60)
        
        # 1. 设置数据
        self.setup_data()
        
        # 2. 创建迁移学习模型
        model = self.create_transfer_model('resnet18', 10)
        # 修改模型输出类别数
        #model = self.create_transfer_model('resnet18', num_classes=2)  # 猫狗是2分类

        
        # 3. 训练模型
        print("\n开始训练...")
        train_losses, test_accuracies = self.train_model(model, epochs=5)
        
        # 4. 最终评估
        final_accuracy = self.evaluate_model(model)
        print(f"\n🎉 迁移学习模型最终准确率: {final_accuracy:.2f}%")
        
        # 5. 比较不同策略
        print("\n" + "="*50)
        print("策略比较")
        print("="*50)
        self.compare_strategies()
        
        # 6. 保存模型
        torch.save(model.state_dict(), 'transfer_learning_model.pth')
        print("\n💾 模型已保存为 'transfer_learning_model.pth'")
        
        # 7. 使用建议
        print("\n💡 迁移学习使用建议:")
        print("• 小数据集(<1000样本): 使用特征提取方法")
        print("• 中等数据集(1000-10000样本): 使用微调方法") 
        print("• 大数据集(>10000样本): 可以考虑从头训练")
        print("• 相似任务: 使用特征提取")
        print("• 不同任务: 使用微调")

def main():
    """主函数"""
    demo = TransferLearningDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()