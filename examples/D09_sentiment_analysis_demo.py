import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import nltk
from nltk.corpus import stopwords
from collections import Counter
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 下载NLTK数据（如果尚未下载）
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class SentimentAnalysisDemo:
    """情感分析实战案例"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        self.stop_words = set(stopwords.words('english'))
        
    def generate_sentiment_data(self):
        """生成模拟的情感分析数据"""
        print("生成模拟情感分析数据...")
        
        # 正面评论模板
        positive_templates = [
            "I love this {product}, it's {adjective}!",
            "This {product} is {adjective}, highly recommended!",
            "Amazing {product}, works perfectly!",
            "Great {product}, very {adjective}!",
            "Excellent {product}, would buy again!",
            "Outstanding {product}, {adjective} quality!",
            "Fantastic {product}, very {adjective}!",
            "Wonderful {product}, exceeds expectations!",
            "Superb {product}, {adjective} performance!",
            "Brilliant {product}, love it!"
        ]
        
        # 负面评论模板
        negative_templates = [
            "I hate this {product}, it's {adjective}!",
            "This {product} is {adjective}, don't buy it!",
            "Terrible {product}, doesn't work at all!",
            "Awful {product}, very {adjective}!",
            "Poor {product}, waste of money!",
            "Disappointing {product}, {adjective} quality!",
            "Horrible {product}, very {adjective}!",
            "Bad {product}, regret buying!",
            "Useless {product}, {adjective} performance!",
            "Worst {product}, avoid it!"
        ]
        
        # 产品列表
        products = ['phone', 'laptop', 'camera', 'headphones', 'tablet', 
                   'watch', 'tv', 'speaker', 'keyboard', 'mouse']
        
        # 形容词列表
        positive_adjectives = ['awesome', 'fantastic', 'excellent', 'great', 'amazing',
                             'outstanding', 'wonderful', 'superb', 'brilliant', 'perfect']
        
        negative_adjectives = ['terrible', 'awful', 'horrible', 'bad', 'poor',
                             'disappointing', 'useless', 'worthless', 'lousy', 'cheap']
        
        # 生成数据
        data = []
        
        # 生成正面评论
        for _ in range(500):
            template = np.random.choice(positive_templates)
            product = np.random.choice(products)
            adjective = np.random.choice(positive_adjectives)
            text = template.format(product=product, adjective=adjective)
            data.append({'text': text, 'label': 1, 'sentiment': 'positive'})
        
        # 生成负面评论
        for _ in range(500):
            template = np.random.choice(negative_templates)
            product = np.random.choice(products)
            adjective = np.random.choice(negative_adjectives)
            text = template.format(product=product, adjective=adjective)
            data.append({'text': text, 'label': 0, 'sentiment': 'negative'})
        
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        # 打乱数据
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"生成数据完成: {len(df)} 条评论")
        print(f"正面评论: {len(df[df['label'] == 1])} 条")
        print(f"负面评论: {len(df[df['label'] == 0])} 条")
        
        return df
    
    def preprocess_text(self, text):
        """文本预处理"""
        # 转换为小写
        text = text.lower()
        
        # 移除标点符号和特殊字符
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # 分词
        words = text.split()
        
        # 移除停用词
        words = [word for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def build_vocabulary(self, texts, max_vocab_size=1000):
        """构建词汇表"""
        print("构建词汇表...")
        
        # 统计词频
        word_freq = Counter()
        for text in texts:
            words = text.split()
            word_freq.update(words)
        
        # 选择最常见的词
        vocab = {word: idx + 2 for idx, (word, _) in 
                enumerate(word_freq.most_common(max_vocab_size - 2))}
        
        # 添加特殊标记
        vocab['<PAD>'] = 0  # 填充标记
        vocab['<UNK>'] = 1  # 未知词标记
        
        print(f"词汇表大小: {len(vocab)}")
        
        return vocab
    
    def text_to_sequence(self, text, vocab, max_length=20):
        """将文本转换为数字序列"""
        words = text.split()
        sequence = [vocab.get(word, vocab['<UNK>']) for word in words]
        
        # 填充或截断
        if len(sequence) < max_length:
            sequence = sequence + [vocab['<PAD>']] * (max_length - len(sequence))
        else:
            sequence = sequence[:max_length]
        
        return sequence
    
    def create_sentiment_model(self, vocab_size, embedding_dim=50, hidden_dim=64):
        """创建情感分析模型"""
        class SentimentModel(nn.Module):
            def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
                super().__init__()
                # 词嵌入层
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                
                # LSTM层
                self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
                
                # 全连接层
                self.fc1 = nn.Linear(hidden_dim * 2, 32)  # 双向LSTM所以是2倍
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(0.3)
                self.fc2 = nn.Linear(32, output_dim)
                
            def forward(self, x):
                # 词嵌入
                embedded = self.embedding(x)  # [batch_size, seq_len, embedding_dim]
                
                # LSTM
                lstm_out, (hidden, cell) = self.lstm(embedded)
                
                # 使用最后一个隐藏状态
                hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
                
                # 全连接层
                x = self.relu(self.fc1(hidden))
                x = self.dropout(x)
                x = self.fc2(x)
                
                return x
        
        return SentimentModel(vocab_size, embedding_dim, hidden_dim, 1)
    
    def train_sentiment_model(self, train_loader, val_loader, vocab_size):
        """训练情感分析模型"""
        print("\n开始训练情感分析模型...")
        
        # 创建模型
        model = self.create_sentiment_model(vocab_size).to(self.device)
        
        # 损失函数和优化器
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        
        # 训练记录
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(20):
            # 训练阶段
            model.train()
            epoch_train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_texts, batch_labels in train_loader:
                batch_texts, batch_labels = batch_texts.to(self.device), batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_texts).squeeze(1)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                epoch_train_loss += loss.item()
                
                # 计算准确率
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                train_correct += (predictions == batch_labels).sum().item()
                train_total += batch_labels.size(0)
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            train_accuracy = train_correct / train_total
            
            # 验证阶段
            model.eval()
            epoch_val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch_texts, batch_labels in val_loader:
                    batch_texts, batch_labels = batch_texts.to(self.device), batch_labels.to(self.device)
                    outputs = model(batch_texts).squeeze(1)
                    loss = criterion(outputs, batch_labels)
                    epoch_val_loss += loss.item()
                    
                    predictions = (torch.sigmoid(outputs) > 0.5).float()
                    val_correct += (predictions == batch_labels).sum().item()
                    val_total += batch_labels.size(0)
            
            avg_val_loss = epoch_val_loss / len(val_loader)
            val_accuracy = val_correct / val_total
            
            # 记录数据
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            train_accuracies.append(train_accuracy)
            val_accuracies.append(val_accuracy)
            
            scheduler.step()
            
            print(f'Epoch [{epoch+1}/20], '
                  f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                  f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        
        return model, train_losses, val_losses, train_accuracies, val_accuracies
    
    def analyze_layer_contributions(self, model, vocab, sample_texts):
        """分析各层对预测的贡献"""
        print("\n分析各层贡献...")
        
        # 存储中间结果
        activations = {}
        
        def get_activation(name):
            def hook(model, input, output):
                if isinstance(output, tuple):
                    activations[name] = output[0].detach()
                else:
                    activations[name] = output.detach()
            return hook
        
        # 注册钩子
        hooks = []
        hooks.append(model.embedding.register_forward_hook(get_activation('embedding')))
        hooks.append(model.lstm.register_forward_hook(get_activation('lstm')))
        hooks.append(model.fc1.register_forward_hook(get_activation('fc1')))
        
        # 处理样本文本
        for i, text in enumerate(sample_texts[:3]):  # 分析前3个样本
            print(f"\n样本 {i+1}: '{text}'")
            
            # 预处理和编码
            processed_text = self.preprocess_text(text)
            sequence = self.text_to_sequence(processed_text, vocab)
            sequence_tensor = torch.tensor(sequence).unsqueeze(0).to(self.device)
            
            # 前向传播
            model.eval()
            with torch.no_grad():
                output = model(sequence_tensor)
                prediction = torch.sigmoid(output).item()
                sentiment = "正面" if prediction > 0.5 else "负面"
                confidence = prediction if prediction > 0.5 else 1 - prediction
            
            print(f"预测: {sentiment} (置信度: {confidence:.4f})")
            
            # 分析各层输出
            print("各层输出统计:")
            for layer_name, activation in activations.items():
                if len(activation.shape) == 3:  # 序列数据
                    stats = f"形状: {activation.shape}, 均值: {activation.mean():.4f}, 标准差: {activation.std():.4f}"
                else:
                    stats = f"形状: {activation.shape}, 均值: {activation.mean():.4f}, 标准差: {activation.std():.4f}"
                print(f"  {layer_name}: {stats}")
            
            # 清除当前样本的激活值
            activations.clear()
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
    
    def visualize_training_process(self, train_losses, val_losses, train_accuracies, val_accuracies):
        """可视化训练过程"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 损失曲线
        ax1.plot(train_losses, label='训练损失')
        ax1.plot(val_losses, label='验证损失')
        ax1.set_title('训练和验证损失', fontproperties={'family': 'SimHei'})
        ax1.set_xlabel('Epoch', fontproperties={'family': 'SimHei'})
        ax1.set_ylabel('Loss', fontproperties={'family': 'SimHei'})
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 准确率曲线
        ax2.plot(train_accuracies, label='训练准确率')
        ax2.plot(val_accuracies, label='验证准确率')
        ax2.set_title('训练和验证准确率', fontproperties={'family': 'SimHei'})
        ax2.set_xlabel('Epoch', fontproperties={'family': 'SimHei'})
        ax2.set_ylabel('Accuracy', fontproperties={'family': 'SimHei'})
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 词嵌入可视化（简化版）
        ax3.text(0.5, 0.5, '词嵌入可视化\n(需要真实文本数据)', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12, fontproperties={'family': 'SimHei'})
        ax3.set_title('词嵌入空间', fontproperties={'family': 'SimHei'})
        ax3.set_xlabel('维度1', fontproperties={'family': 'SimHei'})
        ax3.set_ylabel('维度2', fontproperties={'family': 'SimHei'})
        ax3.grid(True, alpha=0.3)
        
        # 混淆矩阵（使用验证集数据）
        ax4.text(0.5, 0.5, '混淆矩阵\n(需要真实预测结果)', 
                ha='center', va='center', transform=ax4.transAxes, fontsize=12, fontproperties={'family': 'SimHei'})
        ax4.set_title('混淆矩阵', fontproperties={'family': 'SimHei'})
        ax4.set_xlabel('预测标签', fontproperties={'family': 'SimHei'})
        ax4.set_ylabel('真实标签', fontproperties={'family': 'SimHei'})
        
        plt.tight_layout()
        # 简化字体设置，避免中文显示问题
        plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.show()
    
    def demo_real_time_sentiment(self, model, vocab):
        """实时情感分析演示"""
        print("\n" + "="*50)
        print("实时情感分析演示")
        print("="*50)
        print("输入评论进行情感分析 (输入 'quit' 退出)")
        
        model.eval()
        
        while True:
            text = input("\n请输入评论: ").strip()
            
            if text.lower() == 'quit':
                break
            
            if not text:
                continue
            
            # 预处理和预测
            processed_text = self.preprocess_text(text)
            sequence = self.text_to_sequence(processed_text, vocab)
            sequence_tensor = torch.tensor(sequence).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = model(sequence_tensor)
                probability = torch.sigmoid(output).item()
                
                if probability > 0.5:
                    sentiment = "正面"
                    confidence = probability
                else:
                    sentiment = "负面"
                    confidence = 1 - probability
                
                print(f"情感: {sentiment}")
                print(f"置信度: {confidence:.4f}")
                print(f"原始概率: {probability:.4f}")
                
                # 可视化置信度
                plt.figure(figsize=(8, 2))
                # 简化字体设置，避免中文显示问题
                plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial', 'sans-serif', 'SimHei']
                plt.rcParams['axes.unicode_minus'] = False
                plt.barh(['负面', '正面'], [1-probability, probability], color=['red', 'green'])
                plt.xlim(0, 1)
                plt.title('情感分析结果')
                plt.xlabel('概率')
                plt.tight_layout()
                plt.show()
    
    def run_demo(self):
        """运行完整的情感分析演示"""
        print("="*60)
        print("情感分析系统演示")
        print("="*60)
        
        # 1. 生成数据
        df = self.generate_sentiment_data()
        
        # 2. 文本预处理
        print("\n进行文本预处理...")
        df['processed_text'] = df['text'].apply(self.preprocess_text)
        
        # 3. 构建词汇表
        vocab = self.build_vocabulary(df['processed_text'])
        
        # 4. 转换为序列
        sequences = df['processed_text'].apply(lambda x: self.text_to_sequence(x, vocab))
        
        # 5. 创建数据集
        class SentimentDataset(Dataset):
            def __init__(self, sequences, labels):
                self.sequences = torch.tensor(sequences.tolist(), dtype=torch.long)
                self.labels = torch.tensor(labels.tolist(), dtype=torch.float32)
                
            def __len__(self):
                return len(self.sequences)
            
            def __getitem__(self, idx):
                return self.sequences[idx], self.labels[idx]
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, df['label'], test_size=0.2, random_state=42
        )
        
        train_dataset = SentimentDataset(X_train, y_train)
        test_dataset = SentimentDataset(X_test, y_test)
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # 6. 训练模型
        vocab_size = len(vocab)
        model, train_losses, val_losses, train_accuracies, val_accuracies = self.train_sentiment_model(
            train_loader, test_loader, vocab_size
        )
        
        # 7. 可视化训练过程
        self.visualize_training_process(train_losses, val_losses, train_accuracies, val_accuracies)
        
        # 8. 分析层贡献
        sample_texts = [
            "I love this phone, it's amazing!",
            "This laptop is terrible, don't buy it!",
            "The camera works perfectly, excellent quality!"
        ]
        self.analyze_layer_contributions(model, vocab, sample_texts)
        
        # 9. 实时情感分析演示
        self.demo_real_time_sentiment(model, vocab)
        
        print("\n" + "="*60)
        print("演示完成！")
        print("="*60)
def main():
    demo = SentimentAnalysisDemo()
    demo.run_demo()

if __name__ == "__main__":
    main()
