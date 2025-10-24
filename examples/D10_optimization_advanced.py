"""
优化算法进阶演示
对应PPT：幻灯片2-4（动量法、自适应学习率、优化算法对比）
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class OptimizationAdvancedDemo:
    """优化算法进阶演示类"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
    
    def generate_complex_data(self):
        """生成复杂分类数据用于优化算法测试"""
        print("生成复杂分类数据...")
        
        # 生成非线性可分的复杂数据
        X, y = make_classification(
            n_samples=2000,
            n_features=20,           # 20个特征，增加复杂度
            n_informative=15,        # 15个有效特征
            n_redundant=5,           # 5个冗余特征
            n_clusters_per_class=2,  # 每个类别2个簇
            flip_y=0.05,             # 5%的噪声
            random_state=42
        )
        
        # 标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # 转换为PyTorch张量
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        print(f"数据形状: X={X.shape}, y={y.shape}")
        return X, y, scaler
    
    def create_test_model(self, input_size, hidden_size=64, output_size=2):
        """创建测试用神经网络模型"""
        class TestModel(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                # 深层网络用于演示优化算法差异
                self.layers = nn.Sequential(
                    nn.Linear(input_size, hidden_size),
                    nn.ReLU(),
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 2, hidden_size // 4),
                    nn.ReLU(),
                    nn.Linear(hidden_size // 4, output_size)
                )
                
            def forward(self, x):
                return self.layers(x)
        
        return TestModel(input_size, hidden_size, output_size)
    
    def demo_momentum_optimization(self):
        """
        动量优化演示
        对应PPT：幻灯片2（动量法：给梯度下降加上"惯性"）
        """
        print("\n" + "="*60)
        print("动量优化算法演示")
        print("="*60)
        
        # 生成数据
        X, y, scaler = self.generate_complex_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X.numpy(), y.numpy(), test_size=0.2, random_state=42
        )
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        
        # 定义不同动量参数的优化器
        optimizers_config = {
            'SGD (无动量)': {
                'optimizer': optim.SGD,
                'params': {'lr': 0.01, 'momentum': 0.0},
                'color': 'blue',
                'losses': [],
                'accuracies': []
            },
            'SGD (动量=0.5)': {
                'optimizer': optim.SGD,
                'params': {'lr': 0.01, 'momentum': 0.5},
                'color': 'green',
                'losses': [],
                'accuracies': []
            },
            'SGD (动量=0.9)': {
                'optimizer': optim.SGD,
                'params': {'lr': 0.01, 'momentum': 0.9},
                'color': 'red',
                'losses': [],
                'accuracies': []
            }
        }
        
        # 为每个优化器训练模型
        input_size = X_train.shape[1]
        results = {}
        
        for opt_name, config in optimizers_config.items():
            print(f"\n训练 {opt_name}...")
            
            # 创建新模型（确保公平比较）
            model = self.create_test_model(input_size).to(self.device)
            optimizer = config['optimizer'](model.parameters(), **config['params'])
            criterion = nn.CrossEntropyLoss()
            
            # 训练模型
            losses = []
            accuracies = []
            
            for epoch in range(100):
                # 训练阶段
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train.to(self.device))
                loss = criterion(outputs, y_train.to(self.device))
                loss.backward()
                optimizer.step()
                
                # 计算准确率
                model.eval()
                with torch.no_grad():
                    train_outputs = model(X_train.to(self.device))
                    train_pred = torch.argmax(train_outputs, dim=1)
                    train_acc = (train_pred == y_train.to(self.device)).float().mean().item()
                
                losses.append(loss.item())
                accuracies.append(train_acc)
                
                if epoch % 25 == 0:
                    print(f'  Epoch [{epoch}/100], Loss: {loss.item():.4f}, Acc: {train_acc:.4f}')
            
            config['losses'] = losses
            config['accuracies'] = accuracies
            
            # 最终评估
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test.to(self.device))
                test_pred = torch.argmax(test_outputs, dim=1)
                test_acc = (test_pred == y_test.to(self.device)).float().mean().item()
            
            results[opt_name] = test_acc
            print(f"  测试准确率: {test_acc:.4f}")
        
        # 可视化结果
        self._plot_optimization_comparison(optimizers_config, "动量优化算法对比")
        
        return results
    
    def demo_adaptive_optimizers(self):
        """
        自适应优化器演示
        对应PPT：幻灯片3（自适应学习率：智能的"步长调节"）
        """
        print("\n" + "="*60)
        print("自适应优化算法演示")
        print("="*60)
        
        # 生成数据
        X, y, scaler = self.generate_complex_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X.numpy(), y.numpy(), test_size=0.2, random_state=42
        )
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        
        # 定义不同的自适应优化器
        adaptive_optimizers = {
            'SGD': {
                'optimizer': optim.SGD,
                'params': {'lr': 0.01},
                'color': 'blue',
                'description': '基础梯度下降'
            },
            'AdaGrad': {
                'optimizer': optim.Adagrad,
                'params': {'lr': 0.01},
                'color': 'green',
                'description': '累积梯度平方，适合稀疏数据'
            },
            'RMSprop': {
                'optimizer': optim.RMSprop,
                'params': {'lr': 0.001},
                'color': 'red',
                'description': '指数加权移动平均'
            },
            'Adam': {
                'optimizer': optim.Adam,
                'params': {'lr': 0.001},
                'color': 'purple',
                'description': '动量 + 自适应学习率'
            }
        }
        
        input_size = X_train.shape[1]
        results = {}
        
        for opt_name, config in adaptive_optimizers.items():
            print(f"\n训练 {opt_name} - {config['description']}...")
            
            # 创建新模型
            model = self.create_test_model(input_size).to(self.device)
            optimizer = config['optimizer'](model.parameters(), **config['params'])
            criterion = nn.CrossEntropyLoss()
            
            # 训练记录
            losses = []
            accuracies = []
            
            for epoch in range(100):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train.to(self.device))
                loss = criterion(outputs, y_train.to(self.device))
                loss.backward()
                optimizer.step()
                
                # 记录训练指标
                model.eval()
                with torch.no_grad():
                    train_outputs = model(X_train.to(self.device))
                    train_pred = torch.argmax(train_outputs, dim=1)
                    train_acc = (train_pred == y_train.to(self.device)).float().mean().item()
                
                losses.append(loss.item())
                accuracies.append(train_acc)
                
                if epoch % 25 == 0:
                    print(f'  Epoch [{epoch}/100], Loss: {loss.item():.4f}, Acc: {train_acc:.4f}')
            
            config['losses'] = losses
            config['accuracies'] = accuracies
            
            # 测试评估
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test.to(self.device))
                test_pred = torch.argmax(test_outputs, dim=1)
                test_acc = (test_pred == y_test.to(self.device)).float().mean().item()
            
            results[opt_name] = test_acc
            print(f"  测试准确率: {test_acc:.4f}")
        
        # 可视化结果
        self._plot_optimization_comparison(adaptive_optimizers, "自适应优化算法对比")
        
        return results
    
    def _plot_optimization_comparison(self, optimizers_config, title):
        """绘制优化算法对比图"""
        # 设置全局字体以支持中文显示
        plt.rcParams['font.family'] = ['SimHei', 'DejaVu Sans', 'Arial', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False
        
        for opt_name, config in optimizers_config.items():
            ax1.plot(config['losses'], color=config['color'], label=opt_name, linewidth=2)
        ax1.set_xlabel('Epoch', fontproperties={'family': 'SimHei'})
        ax1.set_ylabel('Loss', fontproperties={'family': 'SimHei'})
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # 使用对数坐标更好观察差异
        
        # 准确率曲线
        ax2.set_title(f'{title} - 训练准确率', fontproperties={'family': 'SimHei'})
        for opt_name, config in optimizers_config.items():
            ax2.plot(config['accuracies'], color=config['color'], label=opt_name, linewidth=2)
        ax2.set_xlabel('Epoch', fontproperties={'family': 'SimHei'})
        ax2.set_ylabel('Accuracy', fontproperties={'family': 'SimHei'})
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 收敛速度分析（损失下降到0.5以下的epoch）
        ax3.set_title('收敛速度分析')
        convergence_epochs = []
        opt_names = []
        for opt_name, config in optimizers_config.items():
            losses = config['losses']
            # 找到损失首次低于0.5的epoch
            for i, loss in enumerate(losses):
                if loss < 0.5:
                    convergence_epochs.append(i)
                    opt_names.append(opt_name)
                    break
            else:
                convergence_epochs.append(len(losses))
                opt_names.append(opt_name)
        
        bars = ax3.bar(opt_names, convergence_epochs, color=[config['color'] for config in optimizers_config.values()])
        ax3.set_ylabel('收敛所需Epoch数', fontproperties={'family': 'SimHei'})
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 在柱状图上添加数值标签
        for bar, epoch in zip(bars, convergence_epochs):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{epoch}', ha='center', va='bottom', fontproperties={'family': 'SimHei'})
        
        # 最终性能比较
        ax4.set_title('最终测试准确率', fontproperties={'family': 'SimHei'})
        final_accuracies = [optimizers_config[name].get('final_accuracy', 0.8) for name in optimizers_config.keys()]
        bars = ax4.bar(optimizers_config.keys(), final_accuracies, 
                      color=[config['color'] for config in optimizers_config.values()])
        ax4.set_ylabel('测试准确率', fontproperties={'family': 'SimHei'})
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(0, 1)
        
        # 在柱状图上添加数值标签
        for bar, acc in zip(bars, final_accuracies):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.3f}', ha='center', va='bottom')
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        plt.show()
    
    def demo_optimizer_parameter_sensitivity(self):
        """
        优化器超参数敏感性分析
        对应PPT：幻灯片4（优化算法对比实验）
        """
        print("\n" + "="*60)
        print("优化器超参数敏感性分析")
        print("="*60)
        
        # 测试不同学习率对Adam优化器的影响
        learning_rates = [0.1, 0.01, 0.001, 0.0001]
        colors = ['red', 'blue', 'green', 'purple']
        
        # 生成简单数据用于快速测试
        X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.2, random_state=42)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        input_size = X_train.shape[1]
        results = {}
        
        for i, lr in enumerate(learning_rates):
            print(f"\n测试学习率: {lr}")
            
            model = self.create_test_model(input_size, hidden_size=32, output_size=2).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()
            
            losses = []
            accuracies = []
            
            for epoch in range(50):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train.to(self.device))
                loss = criterion(outputs, y_train.to(self.device))
                loss.backward()
                optimizer.step()
                
                # 记录指标
                losses.append(loss.item())
                
                if epoch % 10 == 0:
                    model.eval()
                    with torch.no_grad():
                        train_outputs = model(X_train.to(self.device))
                        train_pred = torch.argmax(train_outputs, dim=1)
                        train_acc = (train_pred == y_train.to(self.device)).float().mean().item()
                    accuracies.append(train_acc)
            
            # 绘制学习曲线
            axes[0].plot(losses, color=colors[i], label=f'LR={lr}', linewidth=2)
            axes[1].plot(range(0, 50, 10), accuracies, color=colors[i], label=f'LR={lr}', 
                        marker='o', linewidth=2)
            
            # 最终评估
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test.to(self.device))
                test_pred = torch.argmax(test_outputs, dim=1)
                test_acc = (test_pred == y_test.to(self.device)).float().mean().item()
            
            results[lr] = test_acc
            print(f"  测试准确率: {test_acc:.4f}")
        
        # 设置图表属性
        axes[0].set_title('不同学习率的训练损失', fontproperties={'family': 'SimHei'})
        axes[0].set_xlabel('Epoch', fontproperties={'family': 'SimHei'})
        axes[0].set_ylabel('Loss', fontproperties={'family': 'SimHei'})
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_yscale('log')
        
        axes[1].set_title('不同学习率的训练准确率', fontproperties={'family': 'SimHei'})
        axes[1].set_xlabel('Epoch', fontproperties={'family': 'SimHei'})
        axes[1].set_ylabel('Accuracy', fontproperties={'family': 'SimHei'})
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 学习率与最终性能的关系
        axes[2].set_title('学习率与最终性能', fontproperties={'family': 'SimHei'})
        axes[2].semilogx(list(results.keys()), list(results.values()), 'o-', linewidth=2, markersize=8, fontproperties={'family': 'SimHei'})
        axes[2].set_xlabel('学习率', fontproperties={'family': 'SimHei'})
        axes[2].set_ylabel('测试准确率', fontproperties={'family': 'SimHei'})
        axes[2].grid(True, alpha=0.3)
        
        # 最佳学习率分析
        best_lr = max(results, key=results.get)
        axes[3].text(0.1, 0.5, f'最佳学习率: {best_lr}\n对应准确率: {results[best_lr]:.4f}', 
                    fontsize=12, transform=axes[3].transAxes, fontproperties={'family': 'SimHei'})
        axes[3].set_title('最佳学习率分析', fontproperties={'family': 'SimHei'})
        axes[3].axis('off')
        
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        plt.show()
        
        print(f"\n分析结果: 学习率 {best_lr} 表现最佳，准确率 {results[best_lr]:.4f}")

# =============================================================================
# 调试代码区域
# 取消注释以下代码块来运行特定演示
# =============================================================================

def debug_momentum_demo():
    """调试：动量优化演示"""
    demo = OptimizationAdvancedDemo()
    results = demo.demo_momentum_optimization()
    print("\n动量优化结果总结:")
    for opt_name, acc in results.items():
        print(f"  {opt_name}: {acc:.4f}")

def debug_adaptive_demo():
    """调试：自适应优化器演示"""
    demo = OptimizationAdvancedDemo()
    results = demo.demo_adaptive_optimizers()
    print("\n自适应优化器结果总结:")
    for opt_name, acc in results.items():
        print(f"  {opt_name}: {acc:.4f}")

def debug_parameter_sensitivity():
    """调试：超参数敏感性分析"""
    demo = OptimizationAdvancedDemo()
    demo.demo_optimizer_parameter_sensitivity()

if __name__ == "__main__":
    # 运行完整的优化算法演示
    demo = OptimizationAdvancedDemo()
    
    print("第五课第一、二学时 - 优化算法进阶演示")
    print("对应PPT: 幻灯片2-4")
    
    # 取消注释下面的行来运行特定演示
    # debug_momentum_demo()        # 运行动量优化演示
    # debug_adaptive_demo()        # 运行自适应优化器演示  
    # debug_parameter_sensitivity() # 运行超参数敏感性分析
    
    # 或者运行所有演示
    demo.demo_momentum_optimization()
    demo.demo_adaptive_optimizers()
    demo.demo_optimizer_parameter_sensitivity()