import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class OptimizationComparison:
    """优化算法对比演示"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
    
    def generate_complex_data(self):
        """生成复杂数据用于优化算法测试"""
        torch.manual_seed(42)
        
        # 生成非线性可分数据
        n_samples = 1000
        X = torch.randn(n_samples, 2)
        
        # 创建圆形分类边界
        radius = 1.5
        y = (X[:, 0]**2 + X[:, 1]**2 < radius**2).float().unsqueeze(1)
        
        # 添加一些噪声
        noise = torch.randn(n_samples, 1) * 0.1
        y = (y + noise > 0.5).float()
        
        return X, y
    
    def create_model(self):
        """创建测试模型"""
        class TestModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(2, 10)
                self.fc2 = nn.Linear(10, 10)
                self.fc3 = nn.Linear(10, 1)
                
            def forward(self, x):
                x = torch.tanh(self.fc1(x))
                x = torch.tanh(self.fc2(x))
                x = torch.sigmoid(self.fc3(x))
                return x
        
        return TestModel()
    
    def compare_optimizers(self):
        """比较不同优化算法"""
        print("\n" + "="*50)
        print("优化算法对比演示")
        print("="*50)
        
        # 生成数据
        X, y = self.generate_complex_data()
        dataset = TensorDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # 定义不同的优化器
        optimizers_config = {
            'SGD': {
                'optimizer': optim.SGD,
                'params': {'lr': 0.1},
                'color': 'blue',
                'losses': []
            },
            'SGD with Momentum': {
                'optimizer': optim.SGD,
                'params': {'lr': 0.1, 'momentum': 0.9},
                'color': 'green',
                'losses': []
            },
            'Adam': {
                'optimizer': optim.Adam,
                'params': {'lr': 0.01},
                'color': 'red',
                'losses': []
            },
            'RMSprop': {
                'optimizer': optim.RMSprop,
                'params': {'lr': 0.01},
                'color': 'purple',
                'losses': []
            }
        }
        
        # 为每个优化器训练模型
        for opt_name, config in optimizers_config.items():
            print(f"\n训练 {opt_name}...")
            
            # 创建新模型
            model = self.create_model().to(self.device)
            optimizer = config['optimizer'](model.parameters(), **config['params'])
            criterion = nn.BCELoss()
            
            # 训练模型
            losses = []
            for epoch in range(100):
                epoch_loss = 0
                for batch_X, batch_y in dataloader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                avg_loss = epoch_loss / len(dataloader)
                losses.append(avg_loss)
                
                if epoch % 25 == 0:
                    print(f'  Epoch [{epoch}/100], Loss: {avg_loss:.4f}')
            
            config['losses'] = losses
        
        # 绘制对比图
        plt.figure(figsize=(12, 5))
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(1, 2, 1)
        for opt_name, config in optimizers_config.items():
            plt.plot(config['losses'], color=config['color'], label=opt_name, linewidth=2)
        
        plt.title('不同优化算法的损失下降曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # 显示最后50个epoch的细节
        for opt_name, config in optimizers_config.items():
            plt.plot(range(50, 100), config['losses'][50:], 
                    color=config['color'], label=opt_name, linewidth=2)
        
        plt.title('后期训练细节 (Epoch 50-100)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 打印最终损失
        print("\n最终损失比较:")
        for opt_name, config in optimizers_config.items():
            final_loss = config['losses'][-1]
            print(f"{opt_name}: {final_loss:.4f}")
    
    def demo_learning_rate_impact(self):
        """演示学习率的影响"""
        print("\n" + "="*50)
        print("学习率影响演示")
        print("="*50)
        
        # 生成简单的二次函数优化问题
        def quadratic_function(x):
            """简单的二次损失函数: f(x) = (x - 2)^2"""
            return (x - 2) ** 2
        
        def quadratic_gradient(x):
            """梯度: f'(x) = 2*(x - 2)"""
            return 2 * (x - 2)
        
        # 测试不同学习率
        learning_rates = [0.01, 0.1, 0.5, 1.0, 1.5]
        colors = ['blue', 'green', 'red', 'purple', 'orange']
        
        plt.figure(figsize=(15, 5))
        
        # 绘制函数曲线
        x_vals = np.linspace(-1, 5, 100)
        y_vals = quadratic_function(x_vals)
        
        plt.subplot(1, 3, 1)
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.plot(x_vals, y_vals, 'k-', linewidth=2, label='f(x) = (x-2)²')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('优化目标函数')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # 梯度下降过程
        plt.subplot(1, 3, 2)
        plt.plot(x_vals, y_vals, 'k-', linewidth=1, alpha=0.5)
        
        for lr, color in zip(learning_rates, colors):
            # 梯度下降
            x = torch.tensor([-1.0], requires_grad=False)  # 初始点
            x_history = [x.item()]
            loss_history = [quadratic_function(x).item()]
            
            for step in range(20):
                grad = quadratic_gradient(x)
                x = x - lr * grad
                x_history.append(x.item())
                loss_history.append(quadratic_function(x).item())
            
            plt.plot(x_history, loss_history, 'o-', color=color, 
                    label=f'LR={lr}', markersize=4)
        
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('不同学习率的梯度下降路径')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 损失下降曲线
        plt.subplot(1, 3, 3)
        for lr, color in zip(learning_rates, colors):
            x = torch.tensor([-1.0], requires_grad=False)
            loss_history = [quadratic_function(x).item()]
            
            for step in range(20):
                grad = quadratic_gradient(x)
                x = x - lr * grad
                loss_history.append(quadratic_function(x).item())
            
            plt.plot(loss_history, 'o-', color=color, label=f'LR={lr}', markersize=4)
        
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.title('不同学习率的损失下降')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # 使用对数坐标更好地观察差异
        
        plt.tight_layout()
        plt.show()
        
        # 分析结果
        print("\n学习率分析:")
        print("LR=0.01: 收敛慢但稳定")
        print("LR=0.1:  收敛速度适中")
        print("LR=0.5:  快速收敛")
        print("LR=1.0:  在最优值附近震荡")
        print("LR=1.5:  发散，无法收敛")
    
    def demo_overfitting_prevention(self):
        """演示过拟合预防方法"""
        print("\n" + "="*50)
        print("过拟合预防方法演示")
        print("="*50)
        
        # 生成带噪声的多项式数据
        torch.manual_seed(42)
        n_samples = 200
        
        # 真实函数: y = sin(2πx) + 噪声
        X = torch.rand(n_samples, 1) * 2 - 1  # [-1, 1]
        y = torch.sin(2 * np.pi * X) + 0.3 * torch.randn(n_samples, 1)
        
        # 划分训练集和测试集
        split_idx = n_samples // 2
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # 定义复杂模型（容易过拟合）
        class ComplexModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(1, 50),
                    nn.ReLU(),
                    nn.Linear(50, 50),
                    nn.ReLU(),
                    nn.Linear(50, 50),
                    nn.ReLU(),
                    nn.Linear(50, 1)
                )
                
            def forward(self, x):
                return self.network(x)
        
        class RegularizedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.network = nn.Sequential(
                    nn.Linear(1, 20),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(20, 20),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(20, 1)
                )
                
            def forward(self, x):
                return self.network(x)
        
        # 训练两个模型
        models = {
            '复杂模型（无正则化）': ComplexModel(),
            '正则化模型': RegularizedModel()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\n训练 {name}...")
            model = model.to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5 if '正则化' in name else 0)
            criterion = nn.MSELoss()
            
            train_losses = []
            test_losses = []
            
            for epoch in range(1000):
                model.train()
                # 训练损失
                optimizer.zero_grad()
                train_pred = model(X_train.to(self.device))
                train_loss = criterion(train_pred, y_train.to(self.device))
                train_loss.backward()
                optimizer.step()
                
                # 测试损失
                model.eval()
                with torch.no_grad():
                    test_pred = model(X_test.to(self.device))
                    test_loss = criterion(test_pred, y_test.to(self.device))
                
                train_losses.append(train_loss.item())
                test_losses.append(test_loss.item())
                
                if epoch % 200 == 0:
                    print(f'  Epoch [{epoch}/1000], Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
            
            results[name] = {
                'train_losses': train_losses,
                'test_losses': test_losses,
                'model': model
            }
        
        # 绘制结果
        plt.figure(figsize=(15, 5))
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        # 损失曲线
        plt.subplot(1, 3, 1)
        for name, result in results.items():
            plt.plot(result['train_losses'], label=f'{name} - Train', linestyle='-')
            plt.plot(result['test_losses'], label=f'{name} - Test', linestyle='--')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('训练和测试损失')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # 拟合曲线
        x_plot = torch.linspace(-1, 1, 100).unsqueeze(1)
        
        plt.subplot(1, 3, 2)
        plt.scatter(X_train.numpy(), y_train.numpy(), alpha=0.6, label='Train Data')
        plt.scatter(X_test.numpy(), y_test.numpy(), alpha=0.6, label='Test Data')
        
        for name, result in results.items():
            model = result['model']
            model.eval()
            with torch.no_grad():
                y_plot = model(x_plot.to(self.device)).cpu()
            plt.plot(x_plot.numpy(), y_plot.numpy(), label=name, linewidth=2)
        
        # 真实函数
        y_true = torch.sin(2 * np.pi * x_plot)
        plt.plot(x_plot.numpy(), y_true.numpy(), 'k--', label='True Function', linewidth=2)
        
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('模型拟合效果')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 泛化差距
        plt.subplot(1, 3, 3)
        for name, result in results.items():
            generalization_gap = [test - train for train, test in 
                                zip(result['train_losses'], result['test_losses'])]
            plt.plot(generalization_gap, label=name)
        
        plt.xlabel('Epoch')
        plt.ylabel('Test Loss - Train Loss')
        plt.title('泛化差距')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 分析结果
        print("\n过拟合分析:")
        for name, result in results.items():
            final_train_loss = result['train_losses'][-1]
            final_test_loss = result['test_losses'][-1]
            gap = final_test_loss - final_train_loss
            print(f"{name}:")
            print(f"  最终训练损失: {final_train_loss:.4f}")
            print(f"  最终测试损失: {final_test_loss:.4f}")
            print(f"  泛化差距: {gap:.4f}")
            if gap > 0.1:
                print("  → 明显过拟合")
            else:
                print("  → 泛化良好")

def main():
    demo = OptimizationComparison()
    
    # 比较优化算法
    demo.compare_optimizers()
    
    # 演示学习率影响
    demo.demo_learning_rate_impact()
    
    # 演示过拟合预防
    demo.demo_overfitting_prevention()

if __name__ == "__main__":
    main()