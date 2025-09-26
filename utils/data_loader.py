import torch
import torchvision
import torchvision.transforms as transforms
import os
import requests
import zipfile
from tqdm import tqdm

def download_file(url, filename):
    """下载文件并显示进度条"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filename, 'wb') as file, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)

def load_mnist_data(batch_size=64):
    """
    加载MNIST手写数字数据集
    返回：训练集和测试集的数据加载器
    """
    print("正在加载MNIST数据集...")
    
    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 下载训练集
    trainset = torchvision.datasets.MNIST(
        root='./data/mnist', 
        train=True,
        download=True, 
        transform=transform
    )
    
    # 下载测试集
    testset = torchvision.datasets.MNIST(
        root='./data/mnist', 
        train=False,
        download=True, 
        transform=transform
    )
    
    # 创建数据加载器
    trainloader = torch.utils.data.DataLoader(
        trainset, 
        batch_size=batch_size,
        shuffle=True, 
        num_workers=2
    )
    
    testloader = torch.utils.data.DataLoader(
        testset, 
        batch_size=batch_size,
        shuffle=False, 
        num_workers=2
    )
    
    print(f"训练集样本数: {len(trainset)}")
    print(f"测试集样本数: {len(testset)}")
    
    return trainloader, testloader

def load_fer2013_data():
    """
    加载FER2013面部表情识别数据集
    需要手动下载并放置到data/fer2013目录
    """
    print("正在加载FER2013数据集...")
    
    # 检查数据集是否存在
    if not os.path.exists('./data/fer2013/fer2013.csv'):
        print("请先下载FER2013数据集并放置到data/fer2013/目录")
        print("下载地址: https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data")
        return None, None
    
    # 这里简化处理，实际需要解析CSV文件
    # 返回占位符数据
    return None, None

class SimpleDataGenerator:
    """简单的数据生成器，用于教学演示"""
    
    @staticmethod
    def generate_linear_data(num_samples=1000):
        """生成线性回归数据"""
        torch.manual_seed(42)
        X = torch.randn(num_samples, 1)
        y = 2 * X + 1 + 0.1 * torch.randn(num_samples, 1)
        return X, y
    
    @staticmethod
    def generate_xor_data():
        """生成XOR问题数据"""
        X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
        y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)
        return X, y