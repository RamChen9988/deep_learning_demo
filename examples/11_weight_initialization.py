"""
权重初始化对比演示
对应PPT：幻灯片5（权重初始化：好的开始是成功的一半）
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class WeightInitializationDemo:
    """权重初始化对比演示类"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
    
    def generate_deep_network_data(self):
        """生成适合深度网络测试的数据"""
        print("生成深度网络测试数据...")
        
        # 生成高维数据，模拟真实场景
        X, y = make_classification(
            n_samples=2000,
            n_features=50,           # 50个特征
            n_informative=30,        # 30个有效特征
            n_redundant=10,          # 10个冗余特征
            n_repeated=10,           # 10个重复特征
            n_classes=3,             # 3分类问题
            n_clusters_per_class=2,  # 每个类别2个簇
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
    
    def create_deep_network(self, input_size, initialization_method='xavier'):
        """
        创建深度网络，应用不同的初始化方法
        对应PPT：幻灯片5中的初始化方法对比表格
        """
        class DeepNetwork(nn.Module):
            def __init__(self, input_size, init_method='xavier'):
                super().__init__()
                
                # 8层深度网络，用于演示初始化的重要性
                self.layers = nn.Sequential(
                    nn.Linear(input_size, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, 128),
                    nn.ReLU(),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 8),
                    nn.ReLU(),
                    nn.Linear(8, 3)  # 3分类输出
                )
                
                # 应用指定的初始化方法
                self._initialize_weights(init_method)
            
            def _initialize_weights(self, method):
                """应用不同的权重初始化方法"""
                for m in self.layers:
                    if isinstance(m, nn.Linear):
                        if method == 'xavier':
                            init.xavier_uniform_(m.weight)
                            print(f"  应用Xavier初始化: {m.weight.shape}")
                        elif method == 'he':
                            init.kaiming_uniform_(m.weight, nonlinearity='relu')
                            print(f"  应用He初始化: {m.weight.shape}")
                        elif method == 'zero':
                            init.constant_(m.weight, 0.0)
                            print(f"  应用全零初始化: {m.weight.shape}")
                        elif method == 'large':
                            init.normal_(m.weight, mean=0.0, std=1.0)  # 过大标准差
                            print(f"  应用大标准差初始化: {m.weight.shape}")
                        elif method == 'small':
                            init.normal_(m.weight, mean=0.0, std=0.01)  # 过小标准差
                            print(f"  应用小标准差初始化: {m.weight.shape}")
                        
                        # 偏置初始化
                        if m.bias is not None:
                            init.constant_(m.bias, 0.0)
            
            def forward(self, x):
                return self.layers(x)
        
        return DeepNetwork(input_size, initialization_method)
    
    def demo_initialization_methods(self):
        """
        不同初始化方法对比演示
        对应PPT：幻灯片5中的初始化方法选择指南
        """
        print("\n" + "="*60)
        print("权重初始化方法对比演示")
        print("="*60)
        
        # 生成数据
        X, y, scaler = self.generate_deep_network_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X.numpy(), y.numpy(), test_size=0.2, random_state=42
        )
        
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_train, dtype=torch.long)
        
        # 定义不同的初始化方法
        initialization_methods = {
            'Xavier初始化': {
                'method': 'xavier',
                'color': 'blue',
                'description': '适合Tanh/Sigmoid，保持输入输出方差一致',
                'losses': [],
                'gradients': []
            },
            'He初始化': {
                'method': 'he', 
                'color': 'green',
                'description': '适合ReLU，考虑ReLU的激活特性',
                'losses': [],
                'gradients': []
            },
            '全零初始化': {
                'method': 'zero',
                'color': 'red',
                'description': '错误方法：导致对称性破坏',
                'losses': [],
                'gradients': []
            },
            '大标准差初始化': {
                'method': 'large',
                'color': 'purple', 
                'description': '错误方法：可能导致梯度爆炸',
                'losses': [],
                'gradients': []
            },
            '小标准差初始化': {
                'method': 'small',
                'color': 'orange',
                'description': '错误方法：可能导致梯度消失',
                'losses': [],
                'gradients': []
            }
        }
        
        input_size = X_train.shape[1]
        results = {}
        
        for init_name, config in initialization_methods.items():
            print(f"\n测试 {init_name}...")
            print(f"  描述: {config['description']}")
            
            # 创建网络并应用初始化
            model = self.create_deep_network(input_size, config['method']).to(self.device)
            optimizer = optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.CrossEntropyLoss()
            
            # 训练并记录
            losses = []
            gradient_norms = []  # 记录梯度范数
            
            for epoch in range(100):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train.to(self.device))
                loss = criterion(outputs, y_train.to(self.device))
                loss.backward()
                
                # 计算梯度范数（用于诊断梯度问题）
                total_norm = 0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                gradient_norms.append(total_norm)
                
                optimizer.step()
                losses.append(loss.item())
                
                if epoch % 25 == 0:
                    print(f'  Epoch [{epoch}/100], Loss: {loss.item():.4f}, Grad Norm: {total_norm:.4f}')
            
            config['losses'] = losses
            config['gradients'] = gradient_norms
            
            # 最终评估
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test.to(self.device))
                test_pred = torch.argmax(test_outputs, dim=1)
                test_acc = (test_pred == y_test.to(self.device)).float().mean().item()
            
            results[init_name] = test_acc
            print(f"  测试准确率: {test_acc:.4f}")
        
        # 可视化结果
        self._plot_initialization_comparison(initialization_methods, results)
        
        return results
    
    def _plot_initialization_comparison(self, methods_config, results):
        """绘制初始化方法对比图"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 训练损失曲线
        ax1.set_title('不同初始化方法的训练损失')
        for method_name, config in methods_config.items():
            ax1.plot(config['losses'], color=config['color'], label=method_name, linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 梯度范数变化
        ax2.set_title('梯度范数变化（诊断梯度问题）')
        for method_name, config in methods_config.items():
            ax2.plot(config['gradients'], color=config['color'], label=method_name, linewidth=2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('梯度范数')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 最终性能比较
        ax3.set_title('不同初始化方法的最终准确率')
        method_names = list(results.keys())
        accuracies = list(results.values())
        bars = ax3.bar(method_names, accuracies, 
                      color=[methods_config[name]['color'] for name in method_names])
        ax3.set_ylabel('测试准确率')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # 在柱状图上添加数值标签
        for bar, acc in zip(bars, accuracies):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # 初始化方法效果分析
        ax4.set_title('初始化方法效果分析')
        analysis_text = []
        
        # 找出最佳和最差方法
        best_method = max(results, key=results.get)
        worst_method = min(results, key=results.get)
        
        analysis_text.append(f'最佳方法: {best_method}')
        analysis_text.append(f'准确率: {results[best_method]:.4f}')
        analysis_text.append('')
        analysis_text.append(f'最差方法: {worst_method}')
        analysis_text.append(f'准确率: {results[worst_method]:.4f}')
        analysis_text.append('')
        analysis_text.append('推荐:')
        analysis_text.append('- ReLU网络: He初始化')
        analysis_text.append('- Tanh/Sigmoid: Xavier初始化')
        analysis_text.append('- 避免: 全零/极端初始化')
        
        ax4.text(0.05, 0.95, '\n'.join(analysis_text), transform=ax4.transAxes,
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax4.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def demo_activation_distribution(self):
        """
        激活值分布演示
        展示不同初始化方法对激活值分布的影响
        """
        print("\n" + "="*60)
        print("激活值分布分析")
        print("="*60)
        
        # 创建测试网络
        class DistributionNet(nn.Module):
            def __init__(self, init_method):
                super().__init__()
                self.fc1 = nn.Linear(100, 50)
                self.fc2 = nn.Linear(50, 25)
                self.fc3 = nn.Linear(25, 10)
                self._initialize_weights(init_method)
                
                # 用于存储激活值
                self.activations = []
            
            def _initialize_weights(self, method):
                """应用初始化方法"""
                if method == 'xavier':
                    init.xavier_uniform_(self.fc1.weight)
                    init.xavier_uniform_(self.fc2.weight)
                    init.xavier_uniform_(self.fc3.weight)
                elif method == 'he':
                    init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
                    init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
                    init.kaiming_uniform_(self.fc3.weight, nonlinearity='relu')
                elif method == 'large':
                    init.normal_(self.fc1.weight, mean=0.0, std=1.0)
                    init.normal_(self.fc2.weight, mean=0.0, std=1.0)
                    init.normal_(self.fc3.weight, mean=0.0, std=1.0)
                
                # 初始化偏置
                init.constant_(self.fc1.bias, 0.0)
                init.constant_(self.fc2.bias, 0.0)
                init.constant_(self.fc3.bias, 0.0)
            
            def forward(self, x):
                # 第一层
                x = self.fc1(x)
                self.activations.append(x.detach().cpu().numpy())  # 记录激活值
                x = torch.relu(x)
                
                # 第二层
                x = self.fc2(x)
                self.activations.append(x.detach().cpu().numpy())
                x = torch.relu(x)
                
                # 第三层
                x = self.fc3(x)
                self.activations.append(x.detach().cpu().numpy())
                
                return x
        
        # 测试不同的初始化方法
        init_methods = ['xavier', 'he', 'large']
        colors = ['blue', 'green', 'red']
        layer_names = ['第一层', '第二层', '第三层']
        
        # 生成随机输入
        torch.manual_seed(42)
        test_input = torch.randn(1000, 100)  # 1000个样本，100个特征
        
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        
        for i, method in enumerate(init_methods):
            print(f"\n分析 {method} 初始化的激活值分布...")
            
            model = DistributionNet(method)
            with torch.no_grad():
                _ = model(test_input)
            
            # 绘制每层的激活值分布
            for j, (layer_act, layer_name) in enumerate(zip(model.activations, layer_names)):
                ax = axes[j, i]
                
                # 绘制直方图
                ax.hist(layer_act.flatten(), bins=50, color=colors[i], alpha=0.7)
                ax.set_title(f'{method} - {layer_name}')
                ax.set_xlabel('激活值')
                ax.set_ylabel('频次')
                ax.grid(True, alpha=0.3)
                
                # 添加统计信息
                mean_val = np.mean(layer_act)
                std_val = np.std(layer_act)
                ax.text(0.05, 0.95, f'均值: {mean_val:.2f}\n标准差: {std_val:.2f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
        
        print("\n激活值分布分析总结:")
        print("Xavier初始化: 激活值分布相对稳定，各层方差相近")
        print("He初始化: 专门为ReLU设计，保持激活值的方差")
        print("大标准差初始化: 激活值分布过宽，可能导致梯度爆炸")

# =============================================================================
# 调试代码区域
# 取消注释以下代码块来运行特定演示
# =============================================================================

def debug_initialization_demo():
    """调试：初始化方法对比演示"""
    demo = WeightInitializationDemo()
    results = demo.demo_initialization_methods()
    print("\n初始化方法结果总结:")
    for method_name, acc in results.items():
        print(f"  {method_name}: {acc:.4f}")

def debug_activation_distribution():
    """调试：激活值分布分析"""
    demo = WeightInitializationDemo()
    demo.demo_activation_distribution()

if __name__ == "__main__":
    # 运行完整的权重初始化演示
    demo = WeightInitializationDemo()
    
    print("第五课第一、二学时 - 权重初始化对比演示")
    print("对应PPT: 幻灯片5")
    
    # 取消注释下面的行来运行特定演示
    # debug_initialization_demo()    # 运行初始化方法对比
    # debug_activation_distribution() # 运行激活值分布分析
    
    # 或者运行所有演示
    demo.demo_initialization_methods()
    demo.demo_activation_distribution()