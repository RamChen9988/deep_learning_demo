import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

class RealTimeDemo:
    """实时演示类"""
    
    def __init__(self):
        self.model = None
        self.cap = None
        
    def load_pretrained_model(self):
        """加载预训练模型（这里使用随机权重作为演示）"""
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from models.simple_nn import SimpleNet
        self.model = SimpleNet()
        print("模型加载完成（演示用随机权重）")
        
    def webcam_digit_demo(self):
        """摄像头实时数字识别演示"""
        print("启动摄像头数字识别演示...")
        print("按 'q' 退出，按 'c' 捕获图像进行识别")
        
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("无法打开摄像头")
            return
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # 显示原始帧
            cv2.imshow('Real-time Digit Recognition - Press q to quit', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                # 捕获图像并进行识别
                self.process_captured_image(frame)
        
        self.cap.release()
        cv2.destroyAllWindows()
    
    def process_captured_image(self, frame):
        """处理捕获的图像"""
        # 转换为灰度图
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 调整大小到28x28（MNIST尺寸）
        resized = cv2.resize(gray, (28, 28))
        
        # 归一化
        normalized = resized / 255.0
        
        # 转换为Tensor
        input_tensor = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # 预测
        if self.model:
            with torch.no_grad():
                output = self.model(input_tensor)
                prediction = torch.argmax(output).item()
                confidence = torch.softmax(output, dim=1)[0][prediction].item()
                
            print(f"预测数字: {prediction}, 置信度: {confidence:.3f}")
            
            # 显示处理后的图像
            plt.figure(figsize=(6, 3))
            # 简化字体设置，避免中文显示问题
            plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            plt.subplot(1, 2, 1)
            plt.imshow(gray, cmap='gray')
            plt.title('原始图像')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(resized, cmap='gray')
            plt.title(f'预测: {prediction} (置信度: {confidence:.3f})')
            plt.axis('off')
            
            plt.tight_layout()
            plt.show()
    
    def interactive_training_demo(self):
        """交互式训练演示"""
        print("\n=== 交互式训练演示 ===")
        print("这个演示将展示神经网络如何通过训练学习模式")
        
        # 使用XOR问题作为例子
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from utils.data_loader import SimpleDataGenerator
        X, y = SimpleDataGenerator.generate_xor_data()
        
        # 创建模型
        model = nn.Sequential(
            nn.Linear(2, 4),
            nn.Sigmoid(),
            nn.Linear(4, 1),
            nn.Sigmoid()
        )
        
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        
        print("训练前预测:")
        with torch.no_grad():
            before_train = model(X)
            for i in range(4):
                print(f"输入: {X[i].numpy()}, 真实: {y[i].item():.0f}, 预测: {before_train[i].item():.3f}")
        
        # 训练
        print("\n开始训练...")
        losses = []
        for epoch in range(1000):
            optimizer.zero_grad()
            output = model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            
            if epoch in [0, 10, 100, 500, 999]:
                print(f'Epoch {epoch+1}: Loss = {loss.item():.4f}')
        
        print("\n训练后预测:")
        with torch.no_grad():
            after_train = model(X)
            for i in range(4):
                print(f"输入: {X[i].numpy()}, 真实: {y[i].item():.0f}, 预测: {after_train[i].item():.3f}")
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 4))
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title('训练损失曲线')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.bar(['训练前', '训练后'], 
                [criterion(before_train, y).item(), criterion(after_train, y).item()])
        plt.title('训练前后损失对比')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    demo = RealTimeDemo()
    demo.load_pretrained_model()
    
    # 运行交互式演示
    demo.interactive_training_demo()
    
    # 询问是否启动摄像头演示
    choice = input("\n是否启动摄像头演示？(y/n): ")
    if choice.lower() == 'y':
        demo.webcam_digit_demo()

if __name__ == "__main__":
    main()
