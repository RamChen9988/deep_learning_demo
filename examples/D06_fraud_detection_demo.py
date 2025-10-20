import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class FraudDetectionDemo:
    """信用卡欺诈检测案例演示"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
    def generate_fraud_data(self, n_samples=10000, fraud_ratio=0.01):
        """生成模拟的信用卡交易数据"""
        print("生成模拟信用卡交易数据...")
        np.random.seed(42)
        
        # 正常交易模式
        n_normal = int(n_samples * (1 - fraud_ratio))
        n_fraud = n_samples - n_normal
        
        # 生成特征
        data = []
        
        # 正常交易
        for _ in range(n_normal):
            # 正常交易特征
            amount = np.random.lognormal(3, 0.5)  # 交易金额
            hour = np.random.randint(0, 24)       # 交易时间
            location = np.random.choice([0, 1, 2]) # 交易地点
            history = np.random.exponential(10)   # 历史交易频率
            balance = np.random.normal(5000, 1000) # 账户余额
            
            # 正常交易模式
            if hour in [2, 3, 4]:  # 凌晨交易较少
                amount *= 0.3
            if location == 2:  # 国外交易金额较大
                amount *= 1.5
            
            data.append([amount, hour, location, history, balance, 0])  # 0表示正常交易
        
        # 欺诈交易
        for _ in range(n_fraud):
            # 欺诈交易特征 - 与正常交易有不同的模式
            amount = np.random.lognormal(4, 1.0)  # 金额通常较大
            hour = np.random.choice([2, 3, 4, 14, 15])  # 特定时间段
            location = np.random.choice([2])  # 经常在国外
            history = np.random.exponential(2)  # 历史交易较少
            balance = np.random.normal(2000, 500)  # 余额较低
            
            data.append([amount, hour, location, history, balance, 1])  # 1表示欺诈交易
        
        # 转换为DataFrame
        columns = ['amount', 'hour', 'location', 'transaction_frequency', 'account_balance', 'is_fraud']
        df = pd.DataFrame(data, columns=columns)
        
        # 打乱数据
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"生成数据完成: {len(df)} 条记录")
        print(f"正常交易: {len(df[df['is_fraud'] == 0])} 条")
        print(f"欺诈交易: {len(df[df['is_fraud'] == 1])} 条")
        print(f"欺诈比例: {len(df[df['is_fraud'] == 1]) / len(df):.3f}")
        
        return df
    
    def prepare_data(self, df):
        """准备训练数据"""
        print("\n准备训练数据...")
        
        # 分离特征和标签
        X = df.drop('is_fraud', axis=1).values
        y = df['is_fraud'].values
        
        # 标准化特征
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 转换为PyTorch张量
        X_train = torch.tensor(X_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
        
        # 创建数据加载器
        train_dataset = TensorDataset(X_train, y_train)
        test_dataset = TensorDataset(X_test, y_test)
        
        # 由于数据不平衡，使用加权采样
        fraud_indices = torch.where(y_train.squeeze() == 1)[0]
        normal_indices = torch.where(y_train.squeeze() == 0)[0]
        
        weights = torch.zeros(len(y_train))
        weights[fraud_indices] = 1.0 / len(fraud_indices)
        weights[normal_indices] = 1.0 / len(normal_indices)
        weights = weights / weights.sum()
        
        sampler = torch.utils.data.WeightedRandomSampler(weights, len(weights))
        
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        return train_loader, test_loader, scaler, X_train, y_train, X_test, y_test
    
    def create_fraud_detection_model(self, input_size):
        """创建欺诈检测模型"""
        class FraudDetectionModel(nn.Module):
            def __init__(self, input_size):
                super().__init__()
                self.fc1 = nn.Linear(input_size, 64)
                self.bn1 = nn.BatchNorm1d(64)
                self.fc2 = nn.Linear(64, 32)
                self.bn2 = nn.BatchNorm1d(32)
                self.fc3 = nn.Linear(32, 16)
                self.fc4 = nn.Linear(16, 1)
                self.dropout = nn.Dropout(0.3)
                
            def forward(self, x):
                x = torch.relu(self.bn1(self.fc1(x)))
                x = self.dropout(x)
                x = torch.relu(self.bn2(self.fc2(x)))
                x = self.dropout(x)
                x = torch.relu(self.fc3(x))
                x = torch.sigmoid(self.fc4(x))
                return x
        
        return FraudDetectionModel(input_size)
    
    def train_model(self, train_loader, test_loader, input_size):
        """训练欺诈检测模型"""
        print("\n开始训练欺诈检测模型...")
        
        model = self.create_fraud_detection_model(input_size).to(self.device)
        
        # 使用带权重的损失函数处理不平衡数据
        pos_weight = torch.tensor([10.0]).to(self.device)  # 给欺诈样本更高权重
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        
        train_losses = []
        train_f1_scores = []
        val_f1_scores = []
        
        for epoch in range(50):
            # 训练阶段
            model.train()
            epoch_loss = 0
            all_preds = []
            all_labels = []
            
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                
                # 收集预测结果用于计算F1分数
                preds = (outputs > 0.5).float()
                all_preds.extend(preds.cpu().detach().numpy())
                all_labels.extend(batch_y.cpu().numpy())
            
            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            
            # 计算训练F1分数
            from sklearn.metrics import f1_score
            train_f1 = f1_score(all_labels, all_preds, zero_division=0)
            train_f1_scores.append(train_f1)
            
            # 验证阶段
            model.eval()
            val_preds = []
            val_labels = []
            with torch.no_grad():
                for batch_X, batch_y in test_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    preds = (outputs > 0.5).float()
                    val_preds.extend(preds.cpu().numpy())
                    val_labels.extend(batch_y.cpu().numpy())
            
            val_f1 = f1_score(val_labels, val_preds, zero_division=0)
            val_f1_scores.append(val_f1)
            
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/50], Loss: {avg_loss:.4f}, '
                      f'Train F1: {train_f1:.4f}, Val F1: {val_f1:.4f}')
        
        return model, train_losses, train_f1_scores, val_f1_scores
    
    def evaluate_model(self, model, test_loader, X_test, y_test):
        """全面评估模型性能"""
        print("\n模型评估...")
        
        model.eval()
        all_preds = []
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X)
                probs = outputs.cpu().numpy()
                preds = (outputs > 0.5).float().cpu().numpy()
                
                all_probs.extend(probs)
                all_preds.extend(preds)
                all_labels.extend(batch_y.cpu().numpy())
        
        # 转换为numpy数组
        all_probs = np.array(all_probs).flatten()
        all_preds = np.array(all_preds).flatten()
        all_labels = np.array(all_labels).flatten()
        
        # 分类报告
        print("\n分类报告:")
        print(classification_report(all_labels, all_preds, 
                                  target_names=['正常交易', '欺诈交易']))
        
        # 混淆矩阵
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(15, 5))
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(1, 3, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['预测正常', '预测欺诈'],
                   yticklabels=['实际正常', '实际欺诈'])
        plt.title('混淆矩阵')
        
        # ROC曲线
        plt.subplot(1, 3, 2)
        fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('假正率')
        plt.ylabel('真正率')
        plt.title('ROC曲线')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        # 精确率-召回率曲线
        from sklearn.metrics import precision_recall_curve
        plt.subplot(1, 3, 3)
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        
        plt.plot(recall, precision, color='green', lw=2)
        plt.xlabel('召回率')
        plt.ylabel('精确率')
        plt.title('精确率-召回率曲线')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 业务指标分析
        tn, fp, fn, tp = cm.ravel()
        
        print("\n业务指标分析:")
        print(f"准确率: {(tp + tn) / (tp + tn + fp + fn):.4f}")
        print(f"精确率: {tp / (tp + fp):.4f}")
        print(f"召回率: {tp / (tp + fn):.4f}")
        print(f"F1分数: {2 * tp / (2 * tp + fp + fn):.4f}")
        print(f"AUC: {roc_auc:.4f}")
        
        # 成本效益分析
        fraud_cost = 1000  # 假设每笔欺诈交易平均损失1000元
        review_cost = 50   # 假设每笔人工审核成本50元
        
        total_fraud = tp + fn
        detected_fraud = tp
        false_alarms = fp
        
        savings = detected_fraud * fraud_cost
        costs = (detected_fraud + false_alarms) * review_cost
        net_savings = savings - costs
        
        print(f"\n成本效益分析:")
        print(f"检测到的欺诈交易: {detected_fraud}")
        print(f"漏掉的欺诈交易: {fn}")
        print(f"误报数量: {false_alarms}")
        print(f"欺诈防范收益: {savings}元")
        print(f"审核成本: {costs}元")
        print(f"净收益: {net_savings}元")
        
        return all_probs, all_preds, all_labels
    
    def feature_importance_analysis(self, model, feature_names):
        """分析特征重要性"""
        print("\n特征重要性分析...")
        
        # 获取第一层的权重作为特征重要性指标
        weights = model.fc1.weight.data.cpu().numpy()
        feature_importance = np.mean(np.abs(weights), axis=0)
        
        # 绘制特征重要性
        plt.figure(figsize=(10, 6))
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        indices = np.argsort(feature_importance)[::-1]
        
        plt.bar(range(len(feature_importance)), feature_importance[indices])
        plt.xticks(range(len(feature_importance)), 
                  [feature_names[i] for i in indices], rotation=45)
        plt.title('特征重要性分析')
        plt.ylabel('平均权重绝对值')
        plt.tight_layout()
        plt.show()
        
        print("特征重要性排序:")
        for i, idx in enumerate(indices):
            print(f"{i+1}. {feature_names[idx]}: {feature_importance[idx]:.4f}")
    
    def run_demo(self):
        """运行完整的欺诈检测演示"""
        print("="*60)
        print("信用卡欺诈检测系统演示")
        print("="*60)
        
        # 1. 生成数据
        df = self.generate_fraud_data(n_samples=10000, fraud_ratio=0.01)
        
        # 2. 准备数据
        feature_names = ['交易金额', '交易时间', '交易地点', '交易频率', '账户余额']
        train_loader, test_loader, scaler, X_train, y_train, X_test, y_test = self.prepare_data(df)
        
        # 3. 训练模型
        input_size = X_train.shape[1]
        model, train_losses, train_f1, val_f1 = self.train_model(train_loader, test_loader, input_size)
        
        # 4. 绘制训练历史
        plt.figure(figsize=(12, 4))
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.subplot(1, 2, 1)
        plt.plot(train_losses)
        plt.title('训练损失')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        plt.plot(train_f1, label='Train F1')
        plt.plot(val_f1, label='Val F1')
        plt.title('F1分数')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 5. 评估模型
        all_probs, all_preds, all_labels = self.evaluate_model(model, test_loader, X_test, y_test)
        
        # 6. 特征重要性分析
        self.feature_importance_analysis(model, feature_names)
        
        print("\n" + "="*60)
        print("演示完成！")
        print("="*60)

def main():
    demo = FraudDetectionDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()