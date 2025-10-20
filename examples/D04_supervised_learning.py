import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.visualization import plot_training_history

class SupervisedLearningDemo:
    """监督学习完整流程演示"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
    
    def generate_regression_data(self, n_samples=1000):
        """生成回归问题数据"""
        print("生成回归数据...")
        torch.manual_seed(42)
        
        # 生成特征
        X = torch.randn(n_samples, 3)  # 3个特征
        # 生成目标值: y = 2*x1 + 3*x2 - 1.5*x3 + 噪声
        true_weights = torch.tensor([2.0, 3.0, -1.5])
        y = X @ true_weights + 0.1 * torch.randn(n_samples)
        
        # 标准化
        scaler = StandardScaler()
        X_np = scaler.fit_transform(X.numpy())
        X = torch.tensor(X_np, dtype=torch.float32)
        
        return X, y, scaler
    
    def generate_classification_data(self, n_samples=1000):
        """生成分类问题数据"""
        print("生成分类数据...")
        torch.manual_seed(42)
        
        # 生成两个类别的数据
        n_class1 = n_samples // 2
        n_class2 = n_samples - n_class1
        
        # 类别1的数据
        X1 = torch.randn(n_class1, 2) + torch.tensor([2.0, 2.0])
        y1 = torch.zeros(n_class1, 1)
        
        # 类别2的数据
        X2 = torch.randn(n_class2, 2) + torch.tensor([-2.0, -2.0])
        y2 = torch.ones(n_class2, 1)
        
        # 合并数据
        X = torch.cat([X1, X2], dim=0)
        y = torch.cat([y1, y2], dim=0)
        
        return X, y
    
    def demo_regression(self):
        """回归问题演示"""
        print("\n" + "="*50)
        print("回归问题演示 - 房价预测模拟")
        print("="*50)
        
        # 生成数据
        X, y, scaler = self.generate_regression_data(1000)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 定义模型
        class RegressionModel(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.fc1 = nn.Linear(input_size, 64)
                self.fc2 = nn.Linear(64, 32)
                self.fc3 = nn.Linear(32, 1)
                self.dropout = nn.Dropout(0.2)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = torch.relu(self.fc2(x))
                x = self.fc3(x)
                return x
        
        model = RegressionModel(X.shape[1]).to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.MSELoss()  # 均方误差损失
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 训练模型
        print("开始训练回归模型...")
        train_losses = []
        val_losses = []
        
        for epoch in range(100):
            # 训练阶段
            model.train()
            epoch_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_train_loss)
            
            # 验证阶段
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            avg_val_loss = val_loss / len(test_loader)
            val_losses.append(avg_val_loss)
            
            if epoch % 20 == 0:
                print(f'Epoch [{epoch}/100], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # 绘制训练历史
        plt.figure(figsize=(10, 4))
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.title('回归模型训练历史')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 预测结果可视化
        model.eval()
        with torch.no_grad():
            test_pred = model(X_test.to(self.device)).cpu()
        
        plt.subplot(1, 2, 2)
        plt.scatter(y_test.numpy(), test_pred.numpy(), alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title('预测值 vs 真实值')
        plt.xlabel('真实值')
        plt.ylabel('预测值')
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 计算最终指标
        mse = criterion(test_pred, y_test).item()
        print(f'最终测试集MSE: {mse:.4f}')
    
    def demo_classification(self):
        """分类问题演示"""
        print("\n" + "="*50)
        print("分类问题演示 - 客户流失预测模拟")
        print("="*50)
        
        # 生成数据
        X, y = self.generate_classification_data(800)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # 定义模型
        class ClassificationModel(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.fc1 = nn.Linear(input_size, 32)
                self.fc2 = nn.Linear(32, 16)
                self.fc3 = nn.Linear(16, 1)
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = self.dropout(x)
                x = torch.relu(self.fc2(x))
                x = torch.sigmoid(self.fc3(x))
                return x
        
        model = ClassificationModel(X.shape[1]).to(self.device)
        
        # 定义损失函数和优化器
        criterion = nn.BCELoss()  # 二分类交叉熵损失
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        # 训练模型
        print("开始训练分类模型...")
        train_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(100):
            # 训练阶段
            model.train()
            epoch_loss = 0
            correct = 0
            total = 0
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # 计算准确率
                predicted = (outputs > 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            train_accuracy = 100 * correct / total
            
            train_losses.append(avg_train_loss)
            train_accuracies.append(train_accuracy)
            
            # 验证阶段
            model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    predicted = (outputs > 0.5).float()
                    val_total += batch_y.size(0)
                    val_correct += (predicted == batch_y).sum().item()
            
            val_accuracy = 100 * val_correct / val_total
            val_accuracies.append(val_accuracy)
            
            if epoch % 20 == 0:
                print(f'Epoch [{epoch}/100], Loss: {avg_train_loss:.4f}, '
                      f'Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%')
        
        # 绘制训练历史
        plt.figure(figsize=(12, 4))
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('分类模型训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_accuracies, label='Train Accuracy')
        plt.plot(val_accuracies, label='Val Accuracy')
        plt.title('分类模型准确率')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # 最终评估
        model.eval()
        final_val_correct = 0
        final_val_total = 0
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X)
                predicted = (outputs > 0.5).float()
                final_val_total += batch_y.size(0)
                final_val_correct += (predicted == batch_y).sum().item()
        
        final_accuracy = 100 * final_val_correct / final_val_total
        print(f'最终测试集准确率: {final_accuracy:.2f}%')
        
        # 绘制决策边界
        self.plot_decision_boundary(model, X_test, y_test)
    
    def plot_decision_boundary(self, model, X, y):
        """绘制决策边界"""
        model.eval()
        
        # 创建网格
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                           np.linspace(y_min, y_max, 100))
        
        # 预测网格点
        with torch.no_grad():
            mesh_data = torch.tensor(np.c_[xx.ravel(), yy.ravel()], 
                                   dtype=torch.float32).to(self.device)
            Z = model(mesh_data)
            Z = (Z > 0.5).float().cpu().numpy()
            Z = Z.reshape(xx.shape)
        
        # 绘制
        plt.figure(figsize=(8, 6))
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        scatter = plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), 
                            cmap=plt.cm.RdYlBu, edgecolors='k')
        plt.colorbar(scatter)
        plt.title('分类决策边界')
        plt.xlabel('特征 1')
        plt.ylabel('特征 2')
        plt.show()

def main():
    demo = SupervisedLearningDemo()
    
    # 运行回归演示
    demo.demo_regression()
    
    # 运行分类演示
    demo.demo_classification()

if __name__ == "__main__":
    main()