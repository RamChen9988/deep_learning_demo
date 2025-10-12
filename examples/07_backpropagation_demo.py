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

from models.manual_layers import ManualLinear, ManualReLU, ManualSigmoid, ManualSoftmax
from utils.gradient_utils import GradientChecker

class BackpropagationDemo:
    """反向传播手动实现演示"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
    
    def generate_toy_data(self):
        """生成简单的二分类数据"""
        print("生成二分类数据...")
        X, y = make_classification(
            n_samples=1000, 
            n_features=2, 
            n_redundant=0, 
            n_informative=2,
            n_clusters_per_class=1,
            random_state=42
        )
        
        # 标准化
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # 转换为PyTorch张量
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        
        return X, y, scaler
    
    def demo_manual_layers(self):
        """演示手动实现的各层"""
        print("\n" + "="*50)
        print("手动实现神经网络层演示")
        print("="*50)
        
        # 生成测试数据
        torch.manual_seed(42)
        batch_size = 4
        input_size = 3
        hidden_size = 2
        
        # 测试数据
        X = torch.randn(batch_size, input_size)
        print(f"输入数据形状: {X.shape}")
        print(f"输入数据:\n{X}")
        
        # 测试Linear层
        print("\n1. Linear层测试:")
        linear = ManualLinear(input_size, hidden_size)
        print(f"权重形状: {linear.W.shape}")
        print(f"偏置形状: {linear.b.shape}")
        
        # 前向传播
        linear_out = linear.forward(X.numpy())
        print(f"Linear层输出形状: {linear_out.shape}")
        print(f"Linear层输出:\n{linear_out}")
        
        # 测试ReLU层
        print("\n2. ReLU层测试:")
        relu = ManualReLU()
        relu_out = relu.forward(linear_out)
        print(f"ReLU层输出形状: {relu_out.shape}")
        print(f"ReLU层输出:\n{relu_out}")
        
        # 测试Sigmoid层
        print("\n3. Sigmoid层测试:")
        sigmoid = ManualSigmoid()
        sigmoid_out = sigmoid.forward(relu_out)
        print(f"Sigmoid层输出形状: {sigmoid_out.shape}")
        print(f"Sigmoid层输出:\n{sigmoid_out}")
        
        # 测试Softmax层
        print("\n4. Softmax层测试:")
        softmax = ManualSoftmax()
        # 为Softmax生成合适的输入（多分类）
        softmax_input = np.random.randn(batch_size, 3)
        softmax_out = softmax.forward(softmax_input)
        print(f"Softmax层输入形状: {softmax_input.shape}")
        print(f"Softmax层输出形状: {softmax_out.shape}")
        print(f"Softmax层输出(每行和为1):\n{softmax_out}")
        print(f"每行和: {np.sum(softmax_out, axis=1)}")
        
        return linear, relu, sigmoid, softmax
    
    def demo_backpropagation_flow(self):
        """演示反向传播流程"""
        print("\n" + "="*50)
        print("反向传播流程演示")
        print("="*50)
        
        # 构建简单网络
        class ManualTwoLayerNet:
            def __init__(self, input_size, hidden_size, output_size):
                self.linear1 = ManualLinear(input_size, hidden_size)
                self.relu = ManualReLU()
                self.linear2 = ManualLinear(hidden_size, output_size)
                self.sigmoid = ManualSigmoid()
                
            def forward(self, x):
                self.x = x
                self.z1 = self.linear1.forward(x)
                self.a1 = self.relu.forward(self.z1)
                self.z2 = self.linear2.forward(self.a1)
                self.a2 = self.sigmoid.forward(self.z2)
                return self.a2
            
            def backward(self, dout):
                # 反向传播：从输出层到输入层
                dz2 = self.sigmoid.backward(dout)
                da1, dW2, db2 = self.linear2.backward(dz2)
                dz1 = self.relu.backward(da1)
                dx, dW1, db1 = self.linear1.backward(dz1)
                
                return {
                    'dW1': dW1, 'db1': db1,
                    'dW2': dW2, 'db2': db2
                }
            
            def update_parameters(self, grads, learning_rate=0.01):
                # 更新参数
                self.linear1.W -= learning_rate * grads['dW1']
                self.linear1.b -= learning_rate * grads['db1']
                self.linear2.W -= learning_rate * grads['dW2']
                self.linear2.b -= learning_rate * grads['db2']
        
        # 生成数据
        X, y, scaler = self.generate_toy_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X.numpy(), y.numpy(), test_size=0.2, random_state=42
        )
        
        # 创建网络
        input_size = X_train.shape[1]
        hidden_size = 4
        output_size = 1
        
        net = ManualTwoLayerNet(input_size, hidden_size, output_size)
        
        # 训练网络
        print("\n开始训练手动实现的网络...")
        epochs = 100
        losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            
            # 小批量训练
            for i in range(0, len(X_train), 32):
                batch_X = X_train[i:i+32]
                batch_y = y_train[i:i+32]
                
                # 前向传播
                outputs = net.forward(batch_X)
                
                # 计算损失 (二元交叉熵)
                loss = -np.mean(batch_y * np.log(outputs + 1e-8) + 
                               (1 - batch_y) * np.log(1 - outputs + 1e-8))
                epoch_loss += loss
                
                # 计算准确率
                predictions = (outputs > 0.5).astype(float)
                correct += np.sum(predictions == batch_y)
                total += len(batch_y)
                
                # 反向传播
                dout = (outputs - batch_y) / len(batch_y)  # 损失对输出的梯度
                grads = net.backward(dout)
                
                # 更新参数
                net.update_parameters(grads, learning_rate=0.1)
            
            avg_loss = epoch_loss / (len(X_train) // 32)
            accuracy = correct / total
            losses.append(avg_loss)
            
            if epoch % 20 == 0:
                print(f'Epoch [{epoch}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}')
        
        # 绘制训练曲线
        plt.figure(figsize=(10, 4))
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(1, 2, 1)
        plt.plot(losses)
        plt.title('手动实现网络训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        # 测试网络
        test_outputs = net.forward(X_test)
        test_predictions = (test_outputs > 0.5).astype(float)
        test_accuracy = np.mean(test_predictions == y_test)
        
        plt.subplot(1, 2, 2)
        plt.bar(['训练准确率', '测试准确率'], [accuracy, test_accuracy])
        plt.title('模型性能')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"\n最终测试准确率: {test_accuracy:.4f}")
        
        return net
    
    def demo_gradient_checking(self):
        """演示梯度检查"""
        print("\n" + "="*50)
        print("梯度检查演示")
        print("="*50)
        
        # 创建梯度检查器
        checker = GradientChecker()
        
        # 测试Linear层的梯度
        print("测试Linear层梯度...")
        torch.manual_seed(42)
        input_size = 3
        output_size = 2
        batch_size = 4
        
        # 创建手动Linear层和PyTorch Linear层
        manual_linear = ManualLinear(input_size, output_size)
        torch_linear = nn.Linear(input_size, output_size)
        
        # 设置相同的权重
        with torch.no_grad():
            torch_linear.weight.copy_(torch.tensor(manual_linear.W.T))
            torch_linear.bias.copy_(torch.tensor(manual_linear.b))
        
        # 测试数据
        X = torch.randn(batch_size, input_size, requires_grad=True)
        X_np = X.detach().numpy()
        
        # 前向传播
        manual_output = manual_linear.forward(X_np)
        torch_output = torch_linear(X)
        
        # 计算梯度
        dout_np = np.random.randn(*manual_output.shape)
        dout_torch = torch.tensor(dout_np, dtype=torch.float32)
        
        # 手动反向传播
        manual_grads = manual_linear.backward(dout_np)
        
        # PyTorch自动求导
        torch_output.backward(dout_torch)
        
        # 比较梯度
        print("权重梯度比较:")
        print(f"手动计算: {manual_grads['dW'][0, :5]}")  # 显示前5个元素
        print(f"PyTorch计算: {torch_linear.weight.grad[0, :5].numpy()}")
        
        print("\n偏置梯度比较:")
        print(f"手动计算: {manual_grads['db'][:5]}")  # 显示前5个元素
        print(f"PyTorch计算: {torch_linear.bias.grad[:5].numpy()}")
        
        # 计算梯度差异
        weight_diff = np.mean(np.abs(manual_grads['dW'] - torch_linear.weight.grad.numpy().T))
        bias_diff = np.mean(np.abs(manual_grads['db'] - torch_linear.bias.grad.numpy()))
        
        print(f"\n梯度差异:")
        print(f"权重梯度平均差异: {weight_diff:.8f}")
        print(f"偏置梯度平均差异: {bias_diff:.8f}")
        
        if weight_diff < 1e-6 and bias_diff < 1e-6:
            print("✅ 梯度计算正确！")
        else:
            print("❌ 梯度计算有误！")
    
    def compare_manual_vs_autograd(self):
        """比较手动实现与自动求导"""
        print("\n" + "="*50)
        print("手动实现 vs 自动求导比较")
        print("="*50)
        
        # 生成数据
        X, y, scaler = self.generate_toy_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # 手动实现网络
        class ManualNet:
            def __init__(self, input_size, hidden_size, output_size):
                self.linear1 = ManualLinear(input_size, hidden_size)
                self.relu = ManualReLU()
                self.linear2 = ManualLinear(hidden_size, output_size)
                self.sigmoid = ManualSigmoid()
                
            def forward(self, x):
                x = self.linear1.forward(x)
                x = self.relu.forward(x)
                x = self.linear2.forward(x)
                x = self.sigmoid.forward(x)
                return x
            
            def backward(self, dout):
                dout = self.sigmoid.backward(dout)
                dout = self.linear2.backward(dout)
                dout = self.relu.backward(dout)
                dout = self.linear1.backward(dout)
                return dout
        
        # PyTorch自动求导网络
        class TorchNet(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super().__init__()
                self.linear1 = nn.Linear(input_size, hidden_size)
                self.linear2 = nn.Linear(hidden_size, output_size)
                
            def forward(self, x):
                x = torch.relu(self.linear1(x))
                x = torch.sigmoid(self.linear2(x))
                return x
        
        # 训练手动网络
        print("训练手动实现网络...")
        manual_net = ManualNet(2, 4, 1)
        manual_losses = []
        
        for epoch in range(100):
            # 简化训练，只用一个batch
            outputs = manual_net.forward(X_train.numpy())
            loss = -np.mean(y_train.numpy() * np.log(outputs + 1e-8) + 
                           (1 - y_train.numpy()) * np.log(1 - outputs + 1e-8))
            manual_losses.append(loss)
            
            # 反向传播和更新（简化）
            dout = (outputs - y_train.numpy()) / len(X_train)
            manual_net.backward(dout)
        
        # 训练PyTorch网络
        print("训练PyTorch自动求导网络...")
        torch_net = TorchNet(2, 4, 1)
        criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(torch_net.parameters(), lr=0.1)
        
        torch_losses = []
        
        for epoch in range(100):
            optimizer.zero_grad()
            outputs = torch_net(X_train)
            loss = criterion(outputs, y_train)
            loss.backward()
            optimizer.step()
            torch_losses.append(loss.item())
        
        # 比较训练曲线
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(manual_losses, label='手动实现')
        plt.plot(torch_losses, label='PyTorch自动求导')
        plt.title('训练损失对比')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        # 测试性能
        manual_test_out = manual_net.forward(X_test.numpy())
        manual_accuracy = np.mean((manual_test_out > 0.5).astype(float) == y_test.numpy())
        
        torch_net.eval()
        with torch.no_grad():
            torch_test_out = torch_net(X_test)
            torch_accuracy = ((torch_test_out > 0.5).float() == y_test).float().mean().item()
        
        plt.bar(['手动实现', 'PyTorch自动求导'], [manual_accuracy, torch_accuracy])
        plt.title('测试准确率对比')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"手动实现测试准确率: {manual_accuracy:.4f}")
        print(f"PyTorch自动求导测试准确率: {torch_accuracy:.4f}")

def main():
    demo = BackpropagationDemo()
    
    # 演示手动实现的各层
    demo.demo_manual_layers()
    
    # 演示反向传播流程
    demo.demo_backpropagation_flow()
    
    # 演示梯度检查
    demo.demo_gradient_checking()
    
    # 比较手动实现与自动求导
    demo.compare_manual_vs_autograd()

if __name__ == "__main__":
    main()