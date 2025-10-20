"""
优化算法工具函数
包含优化器创建、梯度分析、性能评估等工具
"""
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class OptimizerFactory:
    """优化器工厂类，方便创建和比较不同优化器"""
    
    @staticmethod
    def create_optimizer(model_parameters, optimizer_type, **kwargs):
        """
        创建优化器
        
        Args:
            model_parameters: 模型参数
            optimizer_type: 优化器类型 ('sgd', 'momentum', 'adagrad', 'rmsprop', 'adam')
            **kwargs: 优化器参数
        
        Returns:
            优化器实例
        """
        lr = kwargs.get('lr', 0.001)
        
        if optimizer_type == 'sgd':
            return optim.SGD(model_parameters, lr=lr)
        elif optimizer_type == 'momentum':
            momentum = kwargs.get('momentum', 0.9)
            return optim.SGD(model_parameters, lr=lr, momentum=momentum)
        elif optimizer_type == 'adagrad':
            return optim.Adagrad(model_parameters, lr=lr)
        elif optimizer_type == 'rmsprop':
            return optim.RMSprop(model_parameters, lr=lr)
        elif optimizer_type == 'adam':
            return optim.Adam(model_parameters, lr=lr)
        else:
            raise ValueError(f"不支持的优化器类型: {optimizer_type}")
    
    @staticmethod
    def get_optimizer_configs():
        """获取预定义的优化器配置"""
        return {
            'SGD': {'type': 'sgd', 'lr': 0.01},
            'SGD+Momentum': {'type': 'momentum', 'lr': 0.01, 'momentum': 0.9},
            'AdaGrad': {'type': 'adagrad', 'lr': 0.01},
            'RMSprop': {'type': 'rmsprop', 'lr': 0.001},
            'Adam': {'type': 'adam', 'lr': 0.001}
        }

class GradientAnalyzer:
    """梯度分析工具类"""
    
    def __init__(self):
        self.history = defaultdict(list)
    
    def record_gradients(self, model):
        """记录模型梯度"""
        total_norm = 0
        layer_grads = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                layer_grads[name] = param_norm
        
        total_norm = total_norm ** 0.5
        
        # 记录总梯度范数
        self.history['total_norm'].append(total_norm)
        
        # 记录各层梯度
        for name, norm in layer_grads.items():
            self.history[name].append(norm)
        
        return total_norm, layer_grads
    
    def plot_gradient_history(self, max_layers=5):
        """绘制梯度历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 总梯度范数
        if 'total_norm' in self.history:
            ax1.plot(self.history['total_norm'], linewidth=2)
            ax1.set_title('总梯度范数变化', fontproperties={'family': 'SimHei'})
            ax1.set_xlabel('Step', fontproperties={'family': 'SimHei'})
            ax1.set_ylabel('梯度范数', fontproperties={'family': 'SimHei'})
            ax1.grid(True, alpha=0.3)
            ax1.set_yscale('log')
        
        # 各层梯度（显示前几层）
        layer_count = 0
        for name, grads in self.history.items():
            if name != 'total_norm' and layer_count < max_layers:
                ax2.plot(grads, label=name, linewidth=2)
                layer_count += 1
        
        if layer_count > 0:
            ax2.set_title('各层梯度范数变化')
            ax2.set_xlabel('Step')
            ax2.set_ylabel('梯度范数')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            ax2.set_yscale('log')
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        plt.show()
    
    def analyze_gradient_flow(self):
        """分析梯度流动情况"""
        if not self.history:
            print("没有梯度数据可分析")
            return
        
        print("\n梯度流动分析:")
        print("-" * 30)
        
        # 分析总梯度
        if 'total_norm' in self.history:
            total_grads = self.history['total_norm']
            avg_grad = np.mean(total_grads)
            max_grad = np.max(total_grads)
            min_grad = np.min(total_grads)
            
            print(f"总梯度范数 - 平均: {avg_grad:.4f}, 最大: {max_grad:.4f}, 最小: {min_grad:.4f}")
            
            # 梯度问题诊断
            if max_grad > 1000:
                print("⚠️  检测到可能的梯度爆炸")
            elif min_grad < 1e-6:
                print("⚠️  检测到可能的梯度消失")
            else:
                print("✅ 梯度流动正常")
        
        # 分析层间梯度比例
        layer_grads = {k: v for k, v in self.history.items() if k != 'total_norm'}
        if len(layer_grads) >= 2:
            first_layer = list(layer_grads.keys())[0]
            last_layer = list(layer_grads.keys())[-1]
            
            first_avg = np.mean(layer_grads[first_layer])
            last_avg = np.mean(layer_grads[last_layer])
            
            if first_avg > 0:
                ratio = last_avg / first_avg
                print(f"梯度衰减率 (末层/首层): {ratio:.6f}")
                
                if ratio < 1e-4:
                    print("⚠️  严重梯度消失")
                elif ratio > 100:
                    print("⚠️  严重梯度爆炸")
                else:
                    print("✅ 梯度传播正常")

class TrainingMonitor:
    """训练监控器"""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
    
    def update(self, train_loss, train_acc, val_loss=None, val_acc=None, lr=None):
        """更新训练记录"""
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)
        if val_acc is not None:
            self.history['val_acc'].append(val_acc)
        if lr is not None:
            self.history['learning_rates'].append(lr)
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # 训练损失
        ax1.plot(self.history['train_loss'], label='训练损失', linewidth=2, fontproperties={'family': 'SimHei'})
        if self.history['val_loss']:
            ax1.plot(self.history['val_loss'], label='验证损失', linewidth=2, fontproperties={'family': 'SimHei'})
        ax1.set_title('训练和验证损失', fontproperties={'family': 'SimHei'})
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 训练准确率
        ax2.plot(self.history['train_acc'], label='训练准确率', linewidth=2)
        if self.history['val_acc']:
            ax2.plot(self.history['val_acc'], label='验证准确率', linewidth=2, fontproperties={'family': 'SimHei'})
        ax2.set_title('训练和验证准确率', fontproperties={'family': 'SimHei'})
        ax2.set_xlabel('Epoch', fontproperties={'family': 'SimHei'})
        ax2.set_ylabel('Accuracy', fontproperties={'family': 'SimHei'})
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 学习率变化
        if self.history['learning_rates']:
            ax3.plot(self.history['learning_rates'], linewidth=2, color='purple')
            ax3.set_title('学习率变化', fontproperties={'family': 'SimHei'})
            ax3.set_xlabel('Epoch', fontproperties={'family': 'SimHei'})
            ax3.set_ylabel('Learning Rate', fontproperties={'family': 'SimHei'})
            ax3.grid(True, alpha=0.3)
            ax3.set_yscale('log')
        else:
            ax3.text(0.5, 0.5, '无学习率数据', ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('学习率变化', fontproperties={'family': 'SimHei'})
            ax3.axis('off')
        
        # 训练分析
        ax4.axis('off')
        analysis_text = []
        
        if len(self.history['train_loss']) > 0:
            final_train_loss = self.history['train_loss'][-1]
            final_train_acc = self.history['train_acc'][-1]
            
            analysis_text.append(f'最终训练损失: {final_train_loss:.4f}', fontproperties={'family': 'SimHei'})
            analysis_text.append(f'最终训练准确率: {final_train_acc:.4f}', fontproperties={'family': 'SimHei'})

            if self.history['val_loss']:
                final_val_loss = self.history['val_loss'][-1]
                final_val_acc = self.history['val_acc'][-1]
                analysis_text.append(f'最终验证损失: {final_val_loss:.4f}', fontproperties={'family': 'SimHei'})
                analysis_text.append(f'最终验证准确率: {final_val_acc:.4f}', fontproperties={'family': 'SimHei'})
                
                # 过拟合分析
                overfit_gap = final_train_loss - final_val_loss
                if overfit_gap > 0.1:
                    analysis_text.append('⚠️ 可能欠拟合', fontproperties={'family': 'SimHei'})
                elif overfit_gap < -0.1:
                    analysis_text.append('⚠️ 可能过拟合', fontproperties={'family': 'SimHei'})
                else:
                    analysis_text.append('✅ 拟合程度良好', fontproperties={'family': 'SimHei'})

        if analysis_text:
            ax4.text(0.05, 0.95, '\n'.join(analysis_text), transform=ax4.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.tight_layout()
        plt.show()