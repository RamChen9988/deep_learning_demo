"""
梯度问题诊断演示
对应PPT：幻灯片6（梯度消失与梯度爆炸）
"""
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class GradientProblemsDemo:
    """梯度问题诊断演示类"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
    
    def create_problematic_network(self, depth=10, problem_type='vanishing'):
        """
        创建有梯度问题的深度网络
        对应PPT：幻灯片6中的问题识别部分
        """
        class ProblematicNetwork(nn.Module):
            def __init__(self, input_size, depth, problem_type):
                super().__init__()
                self.depth = depth
                self.problem_type = problem_type
                
                # 创建深度网络
                layers = []
                for i in range(depth):
                    if i == 0:
                        layers.append(nn.Linear(input_size, 50))
                    else:
                        layers.append(nn.Linear(50, 50))
                    
                    # 根据问题类型选择激活函数和初始化
                    if problem_type == 'vanishing':
                        layers.append(nn.Sigmoid())  # Sigmoid容易导致梯度消失
                    elif problem_type == 'exploding':
                        # 使用大权重初始化导致梯度爆炸
                        layers.append(nn.ReLU())
                    else:
                        layers.append(nn.ReLU())
                
                layers.append(nn.Linear(50, 2))  # 输出层
                self.network = nn.Sequential(*layers)
                
                # 应用有问题的初始化
                self._problematic_initialization(problem_type)
                
                # 用于记录梯度
                self.gradients = {f'layer_{i}': [] for i in range(depth + 1)}
            
            def _problematic_initialization(self, problem_type):
                """应用有问题的初始化"""
                for i, layer in enumerate(self.network):
                    if isinstance(layer, nn.Linear):
                        if problem_type == 'vanishing':
                            # 使用Xavier初始化（对Sigmoid合适，但深度网络仍有问题）
                            nn.init.xavier_uniform_(layer.weight)
                        elif problem_type == 'exploding':
                            # 使用大权重导致梯度爆炸
                            nn.init.normal_(layer.weight, mean=0.0, std=2.0)
                        else:
                            # 正常初始化
                            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                        
                        if layer.bias is not None:
                            nn.init.constant_(layer.bias, 0.0)
            
            def forward(self, x):
                return self.network(x)
            
            def record_gradients(self):
                """记录每层的梯度"""
                for i, layer in enumerate(self.network):
                    if isinstance(layer, nn.Linear) and layer.weight.grad is not None:
                        grad_norm = layer.weight.grad.norm().item()
                        self.gradients[f'layer_{i}'].append(grad_norm)
        
        return ProblematicNetwork
    
    def demo_gradient_vanishing(self):
        """
        梯度消失问题演示
        对应PPT：幻灯片6中的梯度消失部分
        """
        print("\n" + "="*60)
        print("梯度消失问题演示")
        print("="*60)
        
        # 生成数据
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.2, random_state=42)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        
        # 创建有梯度消失问题的深度网络
        input_size = X_train.shape[1]
        NetworkClass = self.create_problematic_network(depth=8, problem_type='vanishing')
        model = NetworkClass(input_size, depth=8, problem_type='vanishing').to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        print("训练深度网络（使用Sigmoid激活，容易梯度消失）...")
        
        losses = []
        gradient_records = {f'layer_{i}': [] for i in range(9)}  # 8个隐藏层 + 输出层
        
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train.to(self.device))
            loss = criterion(outputs, y_train.to(self.device))
            loss.backward()
            
            # 记录梯度
            model.record_gradients()
            
            optimizer.step()
            losses.append(loss.item())
            
            if epoch % 20 == 0:
                print(f'  Epoch [{epoch}/100], Loss: {loss.item():.4f}')
        
        # 分析梯度流动
        self._analyze_gradient_flow(model.gradients, "梯度消失问题分析")
        
        return model.gradients, losses
    
    def demo_gradient_exploding(self):
        """
        梯度爆炸问题演示
        对应PPT：幻灯片6中的梯度爆炸部分
        """
        print("\n" + "="*60)
        print("梯度爆炸问题演示")
        print("="*60)
        
        # 生成数据
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.2, random_state=42)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        
        # 创建有梯度爆炸问题的网络
        input_size = X_train.shape[1]
        NetworkClass = self.create_problematic_network(depth=6, problem_type='exploding')
        model = NetworkClass(input_size, depth=6, problem_type='exploding').to(self.device)
        
        # 不使用梯度裁剪，让爆炸问题明显
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        print("训练网络（大权重初始化，容易梯度爆炸）...")
        
        losses = []
        nan_occurred = False
        
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train.to(self.device))
            loss = criterion(outputs, y_train.to(self.device))
            loss.backward()
            
            # 记录梯度
            model.record_gradients()
            
            # 检查梯度是否包含NaN
            for param in model.parameters():
                if param.grad is not None and torch.isnan(param.grad).any():
                    if not nan_occurred:
                        print(f"  ⚠️  第 {epoch} 个epoch检测到梯度NaN（梯度爆炸）")
                        nan_occurred = True
            
            optimizer.step()
            
            current_loss = loss.item()
            if np.isnan(current_loss) or current_loss > 1000:  # 损失异常大
                if not nan_occurred:
                    print(f"  ⚠️  第 {epoch} 个epoch检测到损失异常: {current_loss}")
                    nan_occurred = True
                losses.append(1000)  # 截断异常值
            else:
                losses.append(current_loss)
            
            if epoch % 20 == 0 and not nan_occurred:
                print(f'  Epoch [{epoch}/100], Loss: {current_loss:.4f}')
        
        # 分析梯度流动
        self._analyze_gradient_flow(model.gradients, "梯度爆炸问题分析")
        
        return model.gradients, losses
    
    def demo_gradient_clipping(self):
        """
        梯度裁剪解决方案演示
        对应PPT：幻灯片6中的解决方案部分
        """
        print("\n" + "="*60)
        print("梯度裁剪解决方案演示")
        print("="*60)
        
        # 生成数据
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.2, random_state=42)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.long)
        y_test = torch.tensor(y_test, dtype=torch.long)
        
        # 创建有梯度爆炸问题的网络
        input_size = X_train.shape[1]
        NetworkClass = self.create_problematic_network(depth=6, problem_type='exploding')
        model = NetworkClass(input_size, depth=6, problem_type='exploding').to(self.device)
        
        # 使用梯度裁剪
        optimizer = optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.CrossEntropyLoss()
        
        print("训练网络（应用梯度裁剪解决爆炸问题）...")
        
        losses = []
        clipped_epochs = []  # 记录哪些epoch进行了梯度裁剪
        
        for epoch in range(100):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train.to(self.device))
            loss = criterion(outputs, y_train.to(self.device))
            loss.backward()
            
            # 应用梯度裁剪（核心解决方案）
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # 检查是否进行了裁剪（计算裁剪后的梯度范数）
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            
            if total_norm > 1.0:  # 如果进行了裁剪
                clipped_epochs.append(epoch)
            
            optimizer.step()
            
            # 记录梯度
            model.record_gradients()
            
            losses.append(loss.item())
            
            if epoch % 20 == 0:
                print(f'  Epoch [{epoch}/100], Loss: {loss.item():.4f}, Grad Norm: {total_norm:.4f}')
        
        print(f"  梯度裁剪发生在 {len(clipped_epochs)} 个epoch")
        
        # 分析梯度流动
        self._analyze_gradient_flow(model.gradients, "梯度裁剪效果分析")
        
        return model.gradients, losses, clipped_epochs
    
    def _analyze_gradient_flow(self, gradients, title):
        """分析梯度流动情况"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 梯度随训练的变化（选择前几个epoch查看）
        ax1.set_title(f'{title} - 梯度随时间变化')
        for layer_name, layer_grads in list(gradients.items())[:5]:  # 只显示前5层避免混乱
            if len(layer_grads) > 0:
                ax1.plot(layer_grads[:50], label=layer_name, linewidth=2)  # 只显示前50个epoch
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('梯度范数')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')
        
        # 梯度在层间的传播（选择最后一个epoch的梯度）
        ax2.set_title(f'{title} - 梯度在层间传播')
        layer_indices = []
        final_gradients = []
        
        for layer_name, layer_grads in gradients.items():
            if len(layer_grads) > 0:
                layer_idx = int(layer_name.split('_')[1])
                layer_indices.append(layer_idx)
                final_gradients.append(layer_grads[-1] if len(layer_grads) > 0 else 0)
        
        # 按层索引排序
        sorted_data = sorted(zip(layer_indices, final_gradients))
        layer_indices, final_gradients = zip(*sorted_data)
        
        ax2.plot(layer_indices, final_gradients, 'o-', linewidth=2, markersize=8)
        ax2.set_xlabel('层索引（0=输入层）')
        ax2.set_ylabel('最终梯度范数')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # 添加梯度问题诊断
        if len(final_gradients) > 0:
            gradient_ratio = final_gradients[-1] / final_gradients[0] if final_gradients[0] > 0 else 0
            ax2.text(0.05, 0.95, f'梯度衰减率: {gradient_ratio:.6f}', 
                    transform=ax2.transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            if gradient_ratio < 1e-4:
                diagnosis = "严重梯度消失"
                color = 'red'
            elif gradient_ratio > 100:
                diagnosis = "严重梯度爆炸" 
                color = 'red'
            else:
                diagnosis = "梯度流动正常"
                color = 'green'
            
            ax2.text(0.05, 0.85, f'诊断: {diagnosis}', 
                    transform=ax2.transAxes, fontsize=12, color=color,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def demo_comprehensive_solutions(self):
        """
        综合解决方案演示
        展示解决梯度问题的多种方法
        """
        print("\n" + "="*60)
        print("梯度问题综合解决方案演示")
        print("="*60)
        
        solutions = {
            '问题网络（无解决方案）': {
                'activation': nn.Sigmoid,  # 导致梯度消失
                'init_std': 2.0,          # 导致梯度爆炸
                'clip_grad': False,       # 不裁剪梯度
                'use_batchnorm': False,   # 不使用批量归一化
                'color': 'red'
            },
            '解决方案1（梯度裁剪）': {
                'activation': nn.Sigmoid,
                'init_std': 2.0,
                'clip_grad': True,        # 应用梯度裁剪
                'use_batchnorm': False,
                'color': 'blue'
            },
            '解决方案2（权重初始化）': {
                'activation': nn.Sigmoid,
                'init_std': 0.02,         # 合适的初始化
                'clip_grad': False,
                'use_batchnorm': False,
                'color': 'green'
            },
            '解决方案3（激活函数）': {
                'activation': nn.ReLU,     # 使用ReLU缓解梯度消失
                'init_std': 2.0,
                'clip_grad': False,
                'use_batchnorm': False,
                'color': 'purple'
            },
            '综合解决方案': {
                'activation': nn.ReLU,
                'init_std': 0.02,
                'clip_grad': True,
                'use_batchnorm': True,    # 使用批量归一化
                'color': 'orange'
            }
        }
        
        # 生成数据
        X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        
        X_train, X_test, y_train, y_test = train_test_split(X.numpy(), y.numpy(), test_size=0.2, random_state=42)
        X_train = torch.tensor(X_train, dtype=torch.float32)
        
        results = {}
        
        for sol_name, config in solutions.items():
            print(f"\n测试: {sol_name}")
            
            # 创建网络
            class SolutionNet(nn.Module):
                def __init__(self, input_size, config):
                    super().__init__()
                    self.layers = nn.Sequential(
                        nn.Linear(input_size, 50),
                        config['activation'](),
                        nn.Linear(50, 50),
                        config['activation'](),
                        nn.Linear(50, 50),
                        config['activation'](),
                        nn.Linear(50, 2)
                    )
                    
                    # 应用初始化
                    for layer in self.layers:
                        if isinstance(layer, nn.Linear):
                            nn.init.normal_(layer.weight, mean=0.0, std=config['init_std'])
                
                def forward(self, x):
                    return self.layers(x)
            
            input_size = X_train.shape[1]
            model = SolutionNet(input_size, config).to(self.device)
            optimizer = optim.SGD(model.parameters(), lr=0.01)
            criterion = nn.CrossEntropyLoss()
            
            losses = []
            stable_epochs = 0  # 记录稳定训练的epoch数
            
            for epoch in range(100):
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train.to(self.device))
                loss = criterion(outputs, y.to(self.device))
                loss.backward()
                
                # 应用梯度裁剪
                if config['clip_grad']:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                current_loss = loss.item()
                losses.append(current_loss)
                
                # 检查训练稳定性
                if not (np.isnan(current_loss) or current_loss > 100):
                    stable_epochs += 1
            
            results[sol_name] = {
                'losses': losses,
                'stable_ratio': stable_epochs / 100,
                'color': config['color']
            }
            
            print(f"  稳定训练比例: {stable_epochs / 100:.2f}")
        
        # 可视化解决方案效果
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        for sol_name, result in results.items():
            plt.plot(result['losses'], color=result['color'], label=sol_name, linewidth=2)
        plt.title('不同解决方案的训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        plt.subplot(1, 2, 2)
        names = list(results.keys())
        stable_ratios = [results[name]['stable_ratio'] for name in names]
        colors = [results[name]['color'] for name in names]
        
        bars = plt.bar(names, stable_ratios, color=colors)
        plt.title('训练稳定性比较')
        plt.ylabel('稳定训练比例')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # 添加数值标签
        for bar, ratio in zip(bars, stable_ratios):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{ratio:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        print("\n解决方案效果总结:")
        for sol_name, result in results.items():
            print(f"  {sol_name}: 稳定比例 = {result['stable_ratio']:.2f}")

# =============================================================================
# 调试代码区域
# 取消注释以下代码块来运行特定演示
# =============================================================================

def debug_gradient_vanishing():
    """调试：梯度消失问题演示"""
    demo = GradientProblemsDemo()
    gradients, losses = demo.demo_gradient_vanishing()
    print("梯度消失问题演示完成")

def debug_gradient_exploding():
    """调试：梯度爆炸问题演示"""
    demo = GradientProblemsDemo()
    gradients, losses = demo.demo_gradient_exploding()
    print("梯度爆炸问题演示完成")

def debug_gradient_clipping():
    """调试：梯度裁剪解决方案"""
    demo = GradientProblemsDemo()
    gradients, losses, clipped = demo.demo_gradient_clipping()
    print(f"梯度裁剪演示完成，裁剪发生在 {len(clipped)} 个epoch")

def debug_comprehensive_solutions():
    """调试：综合解决方案"""
    demo = GradientProblemsDemo()
    demo.demo_comprehensive_solutions()

if __name__ == "__main__":
    # 运行完整的梯度问题诊断演示
    demo = GradientProblemsDemo()
    
    print("第五课第一、二学时 - 梯度问题诊断演示")
    print("对应PPT: 幻灯片6")
    
    # 取消注释下面的行来运行特定演示
    # debug_gradient_vanishing()       # 运行梯度消失演示
    # debug_gradient_exploding()       # 运行梯度爆炸演示
    # debug_gradient_clipping()        # 运行梯度裁剪演示
    # debug_comprehensive_solutions()  # 运行综合解决方案演示
    
    # 或者运行所有演示
    demo.demo_gradient_vanishing()
    demo.demo_gradient_exploding() 
    demo.demo_gradient_clipping()
    demo.demo_comprehensive_solutions()