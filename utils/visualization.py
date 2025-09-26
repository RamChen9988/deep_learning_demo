import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid

def plot_images(images, labels, class_names=None, ncols=8):
    """绘制图像网格"""
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    # 将图像转换为网格
    images = images[:ncols*2]  # 显示前16张图片
    grid = make_grid(images, nrow=ncols, normalize=True)
    
    # 绘制
    plt.figure(figsize=(12, 6))
    plt.imshow(grid.permute(1, 2, 0))
    plt.title("样本图像")
    plt.axis('off')
    plt.show()
    
    # 显示对应的标签
    print("对应标签:", [class_names[label] for label in labels[:ncols*2]])

def plot_training_history(train_losses, val_accuracies):
    """绘制训练历史"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 绘制损失曲线
    ax1.plot(train_losses)
    ax1.set_title('训练损失')
    ax1.set_xlabel('迭代次数')
    ax1.set_ylabel('损失')
    ax1.grid(True)
    
    # 绘制准确率曲线
    ax2.plot(val_accuracies)
    ax2.set_title('验证准确率')
    ax2.set_xlabel('迭代次数')
    ax2.set_ylabel('准确率')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def visualize_activations(model, sample_input):
    """可视化激活函数效果"""
    # 获取中间层输出
    activations = {}
    
    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook
    
    # 注册钩子
    model.fc1.register_forward_hook(get_activation('fc1'))
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(sample_input)
    
    # 可视化
    plt.figure(figsize=(10, 4))
    
    # 原始输入
    plt.subplot(1, 2, 1)
    plt.imshow(sample_input.view(28, 28), cmap='gray')
    plt.title('输入图像')
    plt.axis('off')
    
    # 激活值
    plt.subplot(1, 2, 2)
    activation = activations['fc1'][0].numpy()
    plt.bar(range(len(activation)), activation)
    plt.title('第一层激活值')
    plt.xlabel('神经元索引')
    plt.ylabel('激活值')
    
    plt.tight_layout()
    plt.show()

def plot_decision_boundary(model, X, y):
    """绘制决策边界（适用于2D数据）"""
    # 创建网格
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))
    
    # 预测网格点
    model.eval()
    with torch.no_grad():
        Z = model(torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32))
        Z = (Z > 0.5).float().numpy()
        Z = Z.reshape(xx.shape)
    
    # 绘制
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title('决策边界')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.show()