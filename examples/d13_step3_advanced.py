"""
步骤3：高级优化技巧
目标：通过学习率调度和高级优化器进一步提升性能
知识点：学习率调度、优化器比较、早停法
预计准确率：85%+
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
from tqdm import tqdm

plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False

class AdvancedTrainer:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        self.results = {}
    
    def load_data(self):
        """加载带数据增强的数据集"""
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        trainset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform)
        
        self.trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
        self.testloader = DataLoader(testset, batch_size=128, shuffle=False)
        
        return trainset, testset
    
    def create_advanced_model(self):
        """创建更深的网络模型"""
        
        class AdvancedCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    # 卷积块1
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.BatchNorm2d(64),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Dropout(0.3),
                    
                    # 卷积块2
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.BatchNorm2d(128),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Dropout(0.4),
                    
                    # 卷积块3
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Dropout(0.5),
                )
                
                self.classifier = nn.Sequential(
                    nn.Linear(256 * 4 * 4, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(256, 10),
                )
                
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        model = AdvancedCNN().to(self.device)
        print("高级模型特点:")
        print("• 更深的网络结构 (6个卷积层)")
        print("• 每个卷积块包含2个卷积层")
        print("• 递增的Dropout比例")
        print("• 批量归一化应用于所有层")
        
        return model
    
    def test_optimizers(self, model_factory):
        """比较不同优化器的效果"""
        optimizers = {
            'SGD': {
                'optimizer': optim.SGD,
                'params': {'lr': 0.01, 'momentum': 0.9, 'weight_decay': 5e-4}
            },
            'Adam': {
                'optimizer': optim.Adam,
                'params': {'lr': 0.001, 'weight_decay': 1e-4}
            },
            'AdamW': {
                'optimizer': optim.AdamW, 
                'params': {'lr': 0.001, 'weight_decay': 1e-4}
            }
        }
        
        print("\n比较不同优化器效果...")
        
        for name, opt_info in optimizers.items():
            print(f"\n训练使用 {name} 优化器的模型...")
            
            # 创建新模型实例
            model = model_factory()
            optimizer = opt_info['optimizer'](model.parameters(), **opt_info['params'])
            
            # 训练较短时间进行比较
            accuracy = self.train_model(model, optimizer, epochs=20, use_scheduler=False)
            self.results[f'{name}优化器'] = accuracy
            
            print(f"{name} 最终准确率: {accuracy:.2f}%")
    
    def test_schedulers(self, model_factory):
        """比较不同学习率调度器的效果"""
        schedulers = {
            '无调度': None,
            'StepLR': {
                'scheduler': optim.lr_scheduler.StepLR,
                'params': {'step_size': 15, 'gamma': 0.1}
            },
            'CosineAnnealing': {
                'scheduler': optim.lr_scheduler.CosineAnnealingLR,
                'params': {'T_max': 50}
            },
            'ReduceLROnPlateau': {
                'scheduler': optim.lr_scheduler.ReduceLROnPlateau,
                'params': {'mode': 'max', 'patience': 5, 'factor': 0.5}
            }
        }
        
        print("\n比较不同学习率调度器效果...")
        
        for name, sched_info in schedulers.items():
            print(f"\n训练使用 {name} 的模型...")
            
            model = model_factory()
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
            
            if sched_info is None:
                scheduler = None
            else:
                scheduler = sched_info['scheduler'](optimizer, **sched_info['params'])
            
            accuracy = self.train_model(model, optimizer, scheduler, epochs=30)
            self.results[f'{name}调度'] = accuracy
            
            print(f"{name} 最终准确率: {accuracy:.2f}%")
    
    def train_model(self, model, optimizer, scheduler=None, epochs=50, use_scheduler=True):
        """训练模型并返回最终准确率"""
        criterion = nn.CrossEntropyLoss()
        
        # 早停法参数
        best_accuracy = 0.0
        patience = 8
        patience_counter = 0
        
        for epoch in range(epochs):
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0
            
            pbar = tqdm(self.trainloader, desc=f'Epoch {epoch+1}/{epochs}')
            for batch_idx, (inputs, targets) in enumerate(pbar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.3f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
            
            # 评估
            test_acc = self.evaluate_model(model)
            
            # 学习率调度
            if use_scheduler and scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(test_acc)
                else:
                    scheduler.step()
            
            # 早停法检查
            if test_acc > best_accuracy:
                best_accuracy = test_acc
                patience_counter = 0
                torch.save(model.state_dict(), f'best_model_{id(model)}.pth')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f'早停法触发! 最佳准确率: {best_accuracy:.2f}%')
                break
        
        return best_accuracy
    
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
    
    def visualize_comparison(self):
        """可视化比较结果"""
        techniques = list(self.results.keys())
        accuracies = list(self.results.values())
        
        plt.figure(figsize=(12, 6))
        
        # 优化器比较
        opt_results = {k: v for k, v in self.results.items() if '优化器' in k}
        if opt_results:
            plt.subplot(1, 2, 1)
            bars1 = plt.bar(opt_results.keys(), opt_results.values(), color=['lightblue', 'lightgreen', 'orange'])
            plt.title('优化器效果比较', fontweight='bold')
            plt.ylabel('准确率 (%)')
            plt.ylim(0, 100)
            for bar, acc in zip(bars1, opt_results.values()):
                plt.text(bar.get_x() + bar.get_width()/2, acc + 0.5, f'{acc:.1f}%', 
                        ha='center', va='bottom', fontweight='bold')
        
        # 调度器比较
        sched_results = {k: v for k, v in self.results.items() if '调度' in k}
        if sched_results:
            plt.subplot(1, 2, 2)
            bars2 = plt.bar(sched_results.keys(), sched_results.values(), color=['red', 'blue', 'green', 'purple'])
            plt.title('学习率调度器效果比较', fontweight='bold')
            plt.ylabel('准确率 (%)')
            plt.ylim(0, 100)
            plt.xticks(rotation=45)
            for bar, acc in zip(bars2, sched_results.values()):
                plt.text(bar.get_x() + bar.get_width()/2, acc + 0.5, f'{acc:.1f}%', 
                        ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.show()
        
        # 打印总结
        print("\n" + "="*50)
        print("高级优化技术总结")
        print("="*50)
        for technique, acc in self.results.items():
            print(f"• {technique}: {acc:.2f}%")
        
        best_technique = max(self.results, key=self.results.get)
        best_acc = self.results[best_technique]
        print(f"\n最佳组合: {best_technique} - {best_acc:.2f}%")

def main():
    """主函数"""
    print("=" * 60)
    print("步骤3: 高级优化技巧")
    print("目标: 通过学习率调度和优化器选择提升性能")  
    print("=" * 60)
    
    trainer = AdvancedTrainer()
    
    # 加载数据
    trainer.load_data()
    
    # 比较优化器
    trainer.test_optimizers(trainer.create_advanced_model)
    
    # 比较学习率调度器
    trainer.test_schedulers(trainer.create_advanced_model)
    
    # 可视化结果
    trainer.visualize_comparison()
    
    print("\n优化器选择建议:")
    print("• SGD: 需要仔细调参，可能找到更好解")
    print("• Adam: 通常收敛更快，对学习率不敏感") 
    print("• AdamW: 改进的Adam，更适合深度学习")
    
    print("\n学习率调度建议:")
    print("• StepLR: 简单有效，定期降低学习率")
    print("• CosineAnnealing: 平滑变化，理论保证")
    print("• ReduceLROnPlateau: 根据性能自动调整")

if __name__ == "__main__":
    main()