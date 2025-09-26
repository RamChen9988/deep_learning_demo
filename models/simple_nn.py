import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    """简单的全连接神经网络，用于MNIST分类"""
    
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 展平输入
        x = x.view(-1, 28*28)
        
        # 第一层 + ReLU激活
        x = F.relu(self.fc1(x))
        
        # Dropout防止过拟合
        x = self.dropout(x)
        
        # 输出层
        x = self.fc2(x)
        return x

class XORNet(nn.Module):
    """解决XOR问题的神经网络"""
    
    def __init__(self):
        super(XORNet, self).__init__()
        self.fc1 = nn.Linear(2, 4)  # 输入2个特征，隐藏层4个神经元
        self.fc2 = nn.Linear(4, 1)  # 输出1个结果
        
    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))  # 使用sigmoid激活
        x = torch.sigmoid(self.fc2(x))
        return x

class CNNMnist(nn.Module):
    """简单的卷积神经网络，用于MNIST分类"""
    
    def __init__(self):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)  # 输入通道1，输出通道32，卷积核3x3
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        
    def forward(self, x):
        # 第一个卷积层
        x = self.conv1(x)
        x = F.relu(x)
        
        # 第二个卷积层
        x = self.conv2(x)
        x = F.relu(x)
        
        # 最大池化
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        
        # 展平
        x = torch.flatten(x, 1)
        
        # 全连接层
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)

# 演示不同激活函数
class ActivationDemo(nn.Module):
    """演示不同激活函数的模型"""
    
    def __init__(self, activation='relu'):
        super(ActivationDemo, self).__init__()
        self.fc1 = nn.Linear(2, 5)
        self.fc2 = nn.Linear(5, 1)
        self.activation = activation
        
    def forward(self, x):
        x = self.fc1(x)
        
        # 演示不同激活函数
        if self.activation == 'relu':
            x = F.relu(x)
        elif self.activation == 'sigmoid':
            x = torch.sigmoid(x)
        elif self.activation == 'tanh':
            x = torch.tanh(x)
        elif self.activation == 'leaky_relu':
            x = F.leaky_relu(x)
            
        x = self.fc2(x)
        return x