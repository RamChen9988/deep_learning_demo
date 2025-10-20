import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class LayerRelationshipsDemo:
    """神经网络层关系演示"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
    
    def generate_complex_data(self, dataset_type='moons'):
        """生成复杂形状的数据"""
        if dataset_type == 'moons':
            X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)
            title = '半月形数据'
        elif dataset_type == 'circles':
            X, y = make_circles(n_samples=1000, noise=0.1, factor=0.5, random_state=42)
            title = '同心圆数据'
        else:
            X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                                     n_informative=2, random_state=42)
            title = '线性可分数据'
        
        # 标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # 转换为PyTorch张量
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        return X, y, title
    
    def visualize_layer_activations(self):
        """可视化各层激活值"""
        print("\n" + "="*50)
        print("各层激活值可视化")
        print("="*50)
        
        # 创建测试网络
        class DebugNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(2, 10)
                self.linear2 = nn.Linear(10, 10)
                self.linear3 = nn.Linear(10, 2)
                
                # 用于存储中间激活值
                self.activations = {}
                
            def forward(self, x):
                # 注册钩子来捕获激活值
                def get_activation(name):
                    def hook(model, input, output):
                        self.activations[name] = output.detach()
                    return hook
                
                # 为每层注册前向钩子
                self.linear1.register_forward_hook(get_activation('linear1'))
                self.linear2.register_forward_hook(get_activation('linear2'))
                
                # 前向传播
                x = torch.relu(self.linear1(x))
                x = torch.relu(self.linear2(x))
                x = self.linear3(x)
                return x
        
        # 生成数据
        X, y, title = self.generate_complex_data('moons')
        
        # 创建网络
        model = DebugNet()
        
        # 前向传播
        with torch.no_grad():
            outputs = model(X)
        
        # 可视化激活值
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 原始数据
        axes[0, 0].scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', alpha=0.6)
        axes[0, 0].set_title(f'原始数据 - {title}', fontproperties={'family': 'SimHei'})
        axes[0, 0].set_xlabel('特征1', fontproperties={'family': 'SimHei'})
        axes[0, 0].set_ylabel('特征2', fontproperties={'family': 'SimHei'})
        axes[0, 0].grid(True, alpha=0.3)
        
        # 第一层激活值（取前2个神经元）
        activation1 = model.activations['linear1']
        for i in range(2):
            axes[0, 1+i].scatter(X[:, 0], X[:, 1], c=activation1[:, i], cmap='Reds', alpha=0.6)
            axes[0, 1+i].set_title(f'第一层 - 神经元 {i+1} 激活值', fontproperties={'family': 'SimHei'})
            axes[0, 1+i].set_xlabel('特征1', fontproperties={'family': 'SimHei'})
            axes[0, 1+i].set_ylabel('特征2', fontproperties={'family': 'SimHei'})
            axes[0, 1+i].grid(True, alpha=0.3)
        
        # 第二层激活值（取前2个神经元）
        activation2 = model.activations['linear2']
        for i in range(2):
            axes[1, i].scatter(X[:, 0], X[:, 1], c=activation2[:, i], cmap='Blues', alpha=0.6)
            axes[1, i].set_title(f'第二层 - 神经元 {i+1} 激活值', fontproperties={'family': 'SimHei'})
            axes[1, i].set_xlabel('特征1', fontproperties={'family': 'SimHei'})
            axes[1, i].set_ylabel('特征2', fontproperties={'family': 'SimHei'})
            axes[1, i].grid(True, alpha=0.3)
        
        # 激活值分布直方图
        axes[1, 2].hist(activation1.flatten().numpy(), bins=50, alpha=0.7, label='第一层', color='red')
        axes[1, 2].hist(activation2.flatten().numpy(), bins=50, alpha=0.7, label='第二层', color='blue')
        axes[1, 2].set_title('激活值分布', fontproperties={'family': 'SimHei'})
        axes[1, 2].set_xlabel('激活值', fontproperties={'family': 'SimHei'})
        axes[1, 2].set_ylabel('频次', fontproperties={'family': 'SimHei'})
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.show()
        
        print("激活值统计:")
        print(f"第一层激活值 - 均值: {activation1.mean():.4f}, 标准差: {activation1.std():.4f}")
        print(f"第二层激活值 - 均值: {activation2.mean():.4f}, 标准差: {activation2.std():.4f}")
    
    def compare_activation_functions(self):
        """比较不同激活函数的效果"""
        print("\n" + "="*50)
        print("激活函数比较")
        print("="*50)
        
        # 测试不同的激活函数
        activation_functions = {
            'ReLU': nn.ReLU(),
            'Sigmoid': nn.Sigmoid(),
            'Tanh': nn.Tanh(),
            'LeakyReLU': nn.LeakyReLU(0.1)
        }
        
        # 生成测试数据
        x = torch.linspace(-5, 5, 1000).reshape(-1, 1)
        
        # 计算各激活函数的输出和梯度
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        for i, (name, activation) in enumerate(activation_functions.items()):
            # 前向传播
            x.requires_grad_(True)
            y = activation(x)
            
            # 反向传播计算梯度
            y.sum().backward()
            grad = x.grad
            
            # 绘制函数曲线
            axes[i].plot(x.detach().numpy(), y.detach().numpy(), 'b-', linewidth=2, label='函数输出')
            axes[i].plot(x.detach().numpy(), grad.numpy(), 'r--', linewidth=2, label='梯度')
            axes[i].set_title(f'{name}激活函数')
            axes[i].set_xlabel('输入')
            axes[i].set_ylabel('输出/梯度')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(-5, 5)
        
        plt.tight_layout()
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.show()
        
        # 分析各激活函数特性
        print("\n激活函数特性分析:")
        print("ReLU:     计算简单，缓解梯度消失，但可能导致神经元死亡")
        print("Sigmoid:  输出范围(0,1)，适合概率，但容易梯度消失")
        print("Tanh:     输出范围(-1,1)，零中心化，但计算较复杂")
        print("LeakyReLU: 解决ReLU的神经元死亡问题")
    
    def demo_layer_combinations(self):
        """演示不同层组合的效果"""
        print("\n" + "="*50)
        print("不同层组合效果演示")
        print("="*50)
        
        # 定义不同的网络结构
        network_configs = {
            '浅层网络(Linear→ReLU→Linear)': [
                ('Linear', 2, 10),
                ('ReLU',),
                ('Linear', 10, 2)
            ],
            '深层网络(多Linear+ReLU)': [
                ('Linear', 2, 20),
                ('ReLU',),
                ('Linear', 20, 15),
                ('ReLU',),
                ('Linear', 15, 10),
                ('ReLU',),
                ('Linear', 10, 2)
            ],
            '使用Sigmoid': [
                ('Linear', 2, 10),
                ('Sigmoid',),
                ('Linear', 10, 2)
            ],
            '使用Tanh': [
                ('Linear', 2, 10),
                ('Tanh',),
                ('Linear', 10, 2)
            ]
        }
        
        # 生成数据
        X, y, title = self.generate_complex_data('circles')
        
        # 训练不同的网络
        results = {}
        
        for name, layers_config in network_configs.items():
            print(f"\n训练网络: {name}")
            
            # 构建网络
            layers = []
            for config in layers_config:
                if config[0] == 'Linear':
                    layers.append(nn.Linear(config[1], config[2]))
                elif config[0] == 'ReLU':
                    layers.append(nn.ReLU())
                elif config[0] == 'Sigmoid':
                    layers.append(nn.Sigmoid())
                elif config[0] == 'Tanh':
                    layers.append(nn.Tanh())
            
            model = nn.Sequential(*layers)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            # 训练
            losses = []
            for epoch in range(100):
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            # 评估
            with torch.no_grad():
                outputs = model(X)
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions == y).float().mean().item()
            
            results[name] = {
                'losses': losses,
                'accuracy': accuracy,
                'model': model
            }
            
            print(f"最终准确率: {accuracy:.4f}")
        
        # 绘制比较结果
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 训练损失
        axes[0, 0].set_title('训练损失对比')
        for name, result in results.items():
            axes[0, 0].plot(result['losses'], label=name)
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 准确率比较
        names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in names]
        axes[0, 1].bar(names, accuracies)
        axes[0, 1].set_title('测试准确率对比')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 决策边界可视化
        def plot_decision_boundary(model, X, y, ax, title):
            # 创建网格
            x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
            y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                               np.linspace(y_min, y_max, 100))
            
            # 预测
            model.eval()
            with torch.no_grad():
                Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32))
                Z = torch.argmax(Z, dim=1).numpy()
                Z = Z.reshape(xx.shape)
            
            # 绘制
            ax.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
            ax.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
            ax.set_title(title)
            ax.set_xlabel('特征1')
            ax.set_ylabel('特征2')
        
        # 绘制两个网络的决策边界
        plot_decision_boundary(results[names[0]]['model'], X, y, axes[1, 0], f'{names[0]}决策边界')
        plot_decision_boundary(results[names[1]]['model'], X, y, axes[1, 1], f'{names[1]}决策边界')
        
        plt.tight_layout()
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.show()
        
        # 分析结果
        print("\n网络结构分析:")
        best_network = max(results.items(), key=lambda x: x[1]['accuracy'])
        print(f"最佳网络: {best_network[0]} (准确率: {best_network[1]['accuracy']:.4f})")
        
        for name, result in results.items():
            convergence_speed = len(result['losses']) - np.argmin(result['losses'][-10:])
            print(f"{name}: 收敛速度={convergence_speed}, 最终准确率={result['accuracy']:.4f}")
    
    def demo_gradient_flow(self):
        """演示梯度流动"""
        print("\n" + "="*50)
        print("梯度流动演示")
        print("="*50)
        
        # 创建深度网络
        class DeepNet(nn.Module):
            def __init__(self, depth=10):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(2, 10) if i == 0 else
                    nn.Linear(10, 10) if i < depth - 1 else
                    nn.Linear(10, 2)
                    for i in range(depth)
                ])
                self.activations = nn.ModuleList([nn.ReLU() for _ in range(depth-1)])
                
            def forward(self, x):
                for i, layer in enumerate(self.layers):
                    x = layer(x)
                    if i < len(self.activations):
                        x = self.activations[i](x)
                return x
        
        # 生成数据
        X, y, title = self.generate_complex_data('moons')
        
        # 创建不同深度的网络
        depths = [3, 5, 8, 12]
        results = {}
        
        for depth in depths:
            print(f"\n训练深度为 {depth} 的网络...")
            model = DeepNet(depth=depth)
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
            
            # 记录梯度信息
            gradient_norms = []
            
            def hook_fn(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    norm = grad_output[0].norm().item()
                    gradient_norms.append(norm)
            
            # 为每层注册反向钩子
            hooks = []
            for layer in model.layers:
                hook = layer.register_full_backward_hook(hook_fn)
                hooks.append(hook)
            
            # 训练
            losses = []
            for epoch in range(50):
                optimizer.zero_grad()
                outputs = model(X)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            
            # 移除钩子
            for hook in hooks:
                hook.remove()
            
            # 评估
            with torch.no_grad():
                outputs = model(X)
                predictions = torch.argmax(outputs, dim=1)
                accuracy = (predictions == y).float().mean().item()
            
            results[depth] = {
                'losses': losses,
                'accuracy': accuracy,
                'gradient_norms': gradient_norms
            }
        
        # 可视化结果
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 训练损失
        axes[0, 0].set_title('不同深度网络的训练损失')
        for depth, result in results.items():
            axes[0, 0].plot(result['losses'], label=f'深度{depth}')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 准确率
        depths_list = list(results.keys())
        accuracies = [results[d]['accuracy'] for d in depths_list]
        axes[0, 1].bar([str(d) for d in depths_list], accuracies)
        axes[0, 1].set_title('不同深度网络的准确率')
        axes[0, 1].set_xlabel('网络深度')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 梯度范数
        axes[1, 0].set_title('不同层的梯度范数(深度8网络)')
        depth_8_grads = results[8]['gradient_norms']
        layer_indices = list(range(len(depth_8_grads)))
        axes[1, 0].plot(layer_indices, depth_8_grads, 'o-')
        axes[1, 0].set_xlabel('层索引(从输出层到输入层)')
        axes[1, 0].set_ylabel('梯度范数')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_yscale('log')
        
        # 梯度消失分析
        axes[1, 1].set_title('梯度消失现象')
        for depth in [3, 8, 12]:
            grads = results[depth]['gradient_norms']
            if len(grads) > 0:
                # 计算梯度衰减率
                decay_rate = grads[-1] / grads[0] if grads[0] > 0 else 0
                axes[1, 1].bar(str(depth), decay_rate)
        axes[1, 1].set_xlabel('网络深度')
        axes[1, 1].set_ylabel('梯度衰减率(末层/首层)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 分析梯度问题
        print("\n梯度分析:")
        for depth, result in results.items():
            grads = result['gradient_norms']
            if len(grads) > 0:
                decay = grads[-1] / grads[0] if grads[0] > 0 else 0
                print(f"深度{depth}: 梯度衰减率={decay:.6f}, 准确率={result['accuracy']:.4f}")

def main():
    demo = LayerRelationshipsDemo()
    
    # 可视化各层激活值
    demo.visualize_layer_activations()
    
    # 比较激活函数
    demo.compare_activation_functions()
    
    # 演示不同层组合
    demo.demo_layer_combinations()
    
    # 演示梯度流动
    demo.demo_gradient_flow()

if __name__ == "__main__":
    main()
