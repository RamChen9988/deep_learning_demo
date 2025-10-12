import numpy as np
import torch

class GradientChecker:
    """梯度检查工具"""
    
    def __init__(self, epsilon=1e-7):
        self.epsilon = epsilon
    
    def compute_numerical_gradient(self, func, x, dout):
        """计算数值梯度"""
        numerical_grad = np.zeros_like(x)
        
        # 遍历每个元素
        it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
        while not it.finished:
            idx = it.multi_index
            
            # 保存原始值
            original_val = x[idx]
            
            # 计算 f(x + epsilon)
            x[idx] = original_val + self.epsilon
            pos_loss = func(x)
            
            # 计算 f(x - epsilon)
            x[idx] = original_val - self.epsilon
            neg_loss = func(x)
            
            # 计算数值梯度
            numerical_grad[idx] = (pos_loss - neg_loss) / (2 * self.epsilon)
            
            # 恢复原始值
            x[idx] = original_val
            
            it.iternext()
        
        return numerical_grad
    
    def check_gradient(self, manual_grad, numerical_grad, threshold=1e-5):
        """比较手动梯度与数值梯度"""
        # 计算差异
        diff = np.abs(manual_grad - numerical_grad)
        max_diff = np.max(diff)
        avg_diff = np.mean(diff)
        
        print(f"最大梯度差异: {max_diff:.8f}")
        print(f"平均梯度差异: {avg_diff:.8f}")
        
        if max_diff < threshold:
            print("✅ 梯度检查通过！")
            return True
        else:
            print("❌ 梯度检查失败！")
            return False
    
    def visualize_gradient_comparison(self, manual_grad, numerical_grad, layer_name=""):
        """可视化梯度比较"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        

        # 手动梯度
        im1 = axes[0].imshow(manual_grad, cmap='Reds', aspect='auto')
        axes[0].set_title(f'{layer_name} - 手动梯度')
        plt.colorbar(im1, ax=axes[0])
        
        # 数值梯度
        im2 = axes[1].imshow(numerical_grad, cmap='Reds', aspect='auto')
        axes[1].set_title(f'{layer_name} - 数值梯度')
        plt.colorbar(im2, ax=axes[1])
        
        # 差异
        diff = np.abs(manual_grad - numerical_grad)
        im3 = axes[2].imshow(diff, cmap='Reds', aspect='auto')
        axes[2].set_title(f'{layer_name} - 梯度差异')
        plt.colorbar(im3, ax=axes[2])
        
        plt.tight_layout()
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.show()
        
        return diff

class GradientVisualizer:
    """梯度可视化工具"""
    
    @staticmethod
    def plot_gradient_flow(named_parameters):
        """绘制梯度流图"""
        import matplotlib.pyplot as plt
        
        ave_grads = []
        max_grads = []
        layers = []
        
        for n, p in named_parameters:
            if p.requires_grad and "bias" not in n:
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().item())
                max_grads.append(p.grad.abs().max().cpu().item())
        
        plt.figure(figsize=(10, 6))
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.bar(np.arange(len(ave_grads)), ave_grads, alpha=0.7, lw=1, color="b", label='平均梯度')
        plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.7, lw=1, color="r", label='最大梯度')
        plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
        plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
        plt.xlim(left=0, right=len(ave_grads))
        plt.xlabel("层")
        plt.ylabel("梯度值")
        plt.title("梯度流")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def plot_gradient_distribution(named_parameters):
        """绘制梯度分布"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes = axes.flatten()
        
        layer_idx = 0
        for n, p in named_parameters:
            if p.requires_grad and "bias" not in n and layer_idx < 4:
                grad_data = p.grad.cpu().numpy().flatten()
                
                axes[layer_idx].hist(grad_data, bins=50, alpha=0.7)
                axes[layer_idx].set_title(f'{n}梯度分布')
                axes[layer_idx].set_xlabel('梯度值')
                axes[layer_idx].set_ylabel('频次')
                axes[layer_idx].grid(True, alpha=0.3)
                
                # 添加统计信息
                mean_val = np.mean(grad_data)
                std_val = np.std(grad_data)
                axes[layer_idx].text(0.05, 0.95, f'均值: {mean_val:.4f}\n标准差: {std_val:.4f}', 
                                   transform=axes[layer_idx].transAxes, verticalalignment='top')
                
                layer_idx += 1
        
        plt.tight_layout()
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.show()