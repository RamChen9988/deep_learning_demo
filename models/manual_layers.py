import numpy as np

class ManualLinear:
    """手动实现Linear层"""
    
    def __init__(self, input_size, output_size):
        # Xavier初始化
        self.W = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.b = np.zeros(output_size)
        
        # 用于反向传播的缓存
        self.x = None
    
    def forward(self, x):
        """前向传播: y = xW + b"""
        self.x = x
        return np.dot(x, self.W) + self.b
    
    def backward(self, dout):
        """反向传播"""
        # 计算梯度
        dx = np.dot(dout, self.W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)
        
        return dx, dW, db

class ManualReLU:
    """手动实现ReLU层"""
    
    def __init__(self):
        self.mask = None
    
    def forward(self, x):
        """前向传播: y = max(0, x)"""
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    
    def backward(self, dout):
        """反向传播"""
        dout[self.mask] = 0
        return dout

class ManualSigmoid:
    """手动实现Sigmoid层"""
    
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        """前向传播: y = 1 / (1 + exp(-x))"""
        self.out = 1 / (1 + np.exp(-x))
        return self.out
    
    def backward(self, dout):
        """反向传播: dy/dx = y * (1 - y)"""
        dx = dout * self.out * (1 - self.out)
        return dx

class ManualSoftmax:
    """手动实现Softmax层"""
    
    def __init__(self):
        self.out = None
    
    def forward(self, x):
        """前向传播: softmax(x_i) = exp(x_i) / sum(exp(x))"""
        # 数值稳定性处理
        x_shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x = np.exp(x_shifted)
        self.out = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.out
    
    def backward(self, dout):
        """反向传播"""
        # 简化版本，实际使用中通常与交叉熵损失结合
        # 这里返回一个近似的梯度
        batch_size = dout.shape[0]
        dx = self.out.copy()
        return dx / batch_size

class ManualCrossEntropyLoss:
    """手动实现交叉熵损失"""
    
    def __init__(self):
        self.y_pred = None
        self.y_true = None
    
    def forward(self, y_pred, y_true):
        """前向传播"""
        self.y_pred = y_pred
        self.y_true = y_true
        
        # 数值稳定性
        y_pred_clipped = np.clip(y_pred, 1e-8, 1 - 1e-8)
        
        # 计算交叉熵
        if y_true.ndim == 1:  # 类别标签
            batch_size = y_pred.shape[0]
            loss = -np.sum(np.log(y_pred_clipped[np.arange(batch_size), y_true])) / batch_size
        else:  # one-hot编码
            loss = -np.sum(y_true * np.log(y_pred_clipped)) / y_pred.shape[0]
        
        return loss
    
    def backward(self):
        """反向传播"""
        batch_size = self.y_true.shape[0]
        
        if self.y_true.ndim == 1:  # 类别标签
            grad = self.y_pred.copy()
            grad[np.arange(batch_size), self.y_true] -= 1
            grad /= batch_size
        else:  # one-hot编码
            grad = (self.y_pred - self.y_true) / batch_size
        
        return grad