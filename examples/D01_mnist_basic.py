import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.simple_nn import SimpleNet, XORNet
from utils.data_loader import load_mnist_data, SimpleDataGenerator
from utils.visualization import plot_training_history, plot_decision_boundary

def demo_xor_problem():
    """演示XOR问题的解决"""
    print("=== XOR问题演示 ===")
    
    # 生成数据
    X, y = SimpleDataGenerator.generate_xor_data()
    print("XOR数据:")
    print("输入:", X)
    print("输出:", y)
    
    # 创建模型
    model = XORNet()
    criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    # 训练模型
    print("\n训练XOR网络...")
    losses = []
    for epoch in range(1000):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 200 == 0:
            print(f'Epoch [{epoch}/1000], Loss: {loss.item():.4f}')
    
    # 测试模型
    model.eval()
    with torch.no_grad():
        predictions = model(X)
        print("\n预测结果:")
        for i in range(len(X)):
            print(f"输入: {X[i].numpy()}, 真实: {y[i].item():.0f}, 预测: {predictions[i].item():.3f}")

def demo_mnist_classification():
    """演示MNIST手写数字分类"""
    print("\n=== MNIST手写数字分类演示 ===")
    
    # 加载数据
    trainloader, testloader = load_mnist_data(batch_size=64)
    
    # 创建模型
    model = SimpleNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 训练模型（简化版，只训练几个epoch用于演示）
    print("训练模型...")
    train_losses = []
    
    for epoch in range(3):  # 只训练3个epoch用于演示
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader, 0):
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if i % 100 == 99:  # 每100个batch打印一次
                print(f'Epoch [{epoch+1}/3], Batch [{i+1}], Loss: {running_loss/100:.4f}')
                train_losses.append(running_loss/100)
                running_loss = 0.0
    
    # 测试模型
    print("\n测试模型...")
    correct = 0
    total = 0
    model.eval()
    
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    print(f'测试准确率: {accuracy:.2f}%')
    
    # 绘制训练历史
    plot_training_history(train_losses, [accuracy] * len(train_losses))

def demo_activation_functions():
    """演示不同激活函数"""
    print("\n=== 激活函数演示 ===")
    
    # 生成一些测试数据
    x = torch.linspace(-5, 5, 100).reshape(-1, 1)
    
    # 计算不同激活函数的输出
    relu = torch.relu(x)
    sigmoid = torch.sigmoid(x)
    tanh = torch.tanh(x)
    leaky_relu = torch.nn.functional.leaky_relu(x, 0.1)
    
    # 绘制
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 8))
    
    # 简化字体设置，避免中文显示问题
    plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    plt.subplot(2, 2, 1)
    plt.plot(x.numpy(), relu.numpy())
    plt.title('ReLU激活函数')
    plt.grid(True)
    
    plt.subplot(2, 2, 2)
    plt.plot(x.numpy(), sigmoid.numpy())
    plt.title('Sigmoid激活函数')
    plt.grid(True)
    
    plt.subplot(2, 2, 3)
    plt.plot(x.numpy(), tanh.numpy())
    plt.title('Tanh激活函数')
    plt.grid(True)
    
    plt.subplot(2, 2, 4)
    plt.plot(x.numpy(), leaky_relu.numpy())
    plt.title('Leaky ReLU激活函数')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 运行所有演示
    demo_xor_problem()
    demo_activation_functions()
    demo_mnist_classification()