"""
优化相关模型定义
包含各种用于演示优化算法的网络结构
"""
import torch
import torch.nn as nn
import torch.nn.init as init

class DeepTestNetwork(nn.Module):
    """深度测试网络，用于演示优化算法和初始化方法"""
    
    def __init__(self, input_size, hidden_sizes=[128, 256, 128, 64, 32], output_size=2, 
                 activation='relu', initialization='he'):
        super().__init__()
        
        # 构建深度网络层
        layers = []
        prev_size = input_size
        
        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            
            # 选择激活函数
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.1))
            
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # 应用初始化
        self._initialize_weights(initialization)
    
    def _initialize_weights(self, method):
        """应用权重初始化方法"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                if method == 'xavier':
                    init.xavier_uniform_(layer.weight)
                elif method == 'he':
                    init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                elif method == 'zero':
                    init.constant_(layer.weight, 0.0)
                elif method == 'large':
                    init.normal_(layer.weight, mean=0.0, std=2.0)
                elif method == 'small':
                    init.normal_(layer.weight, mean=0.0, std=0.01)
                elif method == 'lecun':
                    init.normal_(layer.weight, mean=0.0, std=1.0 / layer.weight.size(1))
                
                # 偏置初始化
                if layer.bias is not None:
                    init.constant_(layer.bias, 0.0)
    
    def forward(self, x):
        return self.network(x)

class GradientTrackingNetwork(nn.Module):
    """梯度追踪网络，用于记录和分析梯度流动"""
    
    def __init__(self, input_size, layer_sizes=[100, 80, 60, 40, 20], output_size=2):
        super().__init__()
        
        # 构建网络层
        self.layers = nn.ModuleList()
        prev_size = input_size
        
        for size in layer_sizes:
            self.layers.append(nn.Linear(prev_size, size))
            self.layers.append(nn.ReLU())
            prev_size = size
        
        self.output_layer = nn.Linear(prev_size, output_size)
        
        # 梯度记录
        self.gradient_records = {}
        self._register_backward_hooks()
    
    def _register_backward_hooks(self):
        """注册反向传播钩子来记录梯度"""
        def make_hook(name):
            def hook(module, grad_input, grad_output):
                if grad_output[0] is not None:
                    grad_norm = grad_output[0].norm().item()
                    if name not in self.gradient_records:
                        self.gradient_records[name] = []
                    self.gradient_records[name].append(grad_norm)
            return hook
        
        # 为每个线性层注册钩子
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.Linear):
                layer.register_full_backward_hook(make_hook(f'hidden_{i}'))
        
        self.output_layer.register_full_backward_hook(make_hook('output'))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x
    
    def clear_gradients(self):
        """清空梯度记录"""
        self.gradient_records = {}

class OptimizationComparisonModel(nn.Module):
    """优化算法对比专用模型"""
    
    def __init__(self, input_size, complexity='medium'):
        super().__init__()
        
        if complexity == 'simple':
            # 简单网络 - 用于基础演示
            self.network = nn.Sequential(
                nn.Linear(input_size, 50),
                nn.ReLU(),
                nn.Linear(50, 25),
                nn.ReLU(),
                nn.Linear(25, 2)
            )
        elif complexity == 'medium':
            # 中等复杂度 - 用于标准测试
            self.network = nn.Sequential(
                nn.Linear(input_size, 100),
                nn.ReLU(),
                nn.Linear(100, 50),
                nn.ReLU(),
                nn.Linear(50, 25),
                nn.ReLU(),
                nn.Linear(25, 10),
                nn.ReLU(),
                nn.Linear(10, 2)
            )
        else:  # complex
            # 复杂网络 - 用于挑战性测试
            self.network = nn.Sequential(
                nn.Linear(input_size, 200),
                nn.ReLU(),
                nn.Linear(200, 150),
                nn.ReLU(),
                nn.Linear(150, 100),
                nn.ReLU(),
                nn.Linear(100, 80),
                nn.ReLU(),
                nn.Linear(80, 60),
                nn.ReLU(),
                nn.Linear(60, 40),
                nn.ReLU(),
                nn.Linear(40, 20),
                nn.ReLU(),
                nn.Linear(20, 2)
            )
    
    def forward(self, x):
        return self.network(x)