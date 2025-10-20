# 修复的情感分析模型
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight

import jieba
import pickle
import os

# 设置随机种子，保证结果可复现
np.random.seed(42)
tf.random.set_seed(42)

class FixedSentimentAnalysisModel:
    def __init__(self, max_words=5000, max_len=50, embedding_dim=100):
        """
        初始化情感分析模型参数
        
        参数:
            max_words: 最大词汇量（减少过拟合）
            max_len: 句子最大长度（减少计算量）
            embedding_dim: 词嵌入维度
        """
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.label_mapping = None
        self.model = None

    def tokenize_chinese(self, text):
        """中文分词"""
        return ' '.join(jieba.cut(text))
    
    def load_data(self, file_path):
        """
        加载并预处理数据集
        """
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 检查必要的列是否存在
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("数据集必须包含'text'和'label'列")
        
        # 提取文本和标签
        texts = df['text'].astype(str).tolist()
        texts = [self.tokenize_chinese(text) for text in texts]
        labels = df['label'].tolist()
        
        # 创建标签映射
        unique_labels = list(set(labels))
        self.label_mapping = {label: i for i, label in enumerate(unique_labels)}
        
        print(f"发现的情感标签: {unique_labels}")
        print(f"标签映射: {self.label_mapping}")
        print(f"数据集大小: {len(texts)}")
        
        # 统计类别分布
        label_counts = {label: labels.count(label) for label in unique_labels}
        print(f"类别分布: {label_counts}")
        
        # 将标签转换为数字
        labels = [self.label_mapping[label] for label in labels]
        
        # 初始化Tokenizer（使用更小的词汇表）
        self.tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        
        # 将文本转换为序列
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # 对序列进行填充
        X = pad_sequences(sequences, maxlen=self.max_len)
        
        # 将标签转换为独热编码
        y = to_categorical(labels)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=labels
        )
        
        # 计算类别权重（处理类别不平衡）
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(labels),
            y=labels
        )
        self.class_weights = dict(enumerate(class_weights))
        print(f"类别权重: {self.class_weights}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, num_classes):
        """
        构建改进的LSTM情感分析模型
        """
        self.model = Sequential([
            # 嵌入层
            Embedding(
                input_dim=self.max_words,
                output_dim=self.embedding_dim,
                input_length=self.max_len,
                mask_zero=True  # 忽略填充的0
            ),
            # LSTM层（使用更小的单元数）
            LSTM(units=64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
            # BatchNormalization层
            BatchNormalization(),
            # Dropout层
            Dropout(0.5),
            # 输出层
            Dense(units=num_classes, activation='softmax')
        ])
        
        # 编译模型（使用更低的学习率）
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("模型结构:")
        self.model.summary()
    
    def train(self, X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
        """
        训练模型
        """
        num_classes = y_train.shape[1]
        
        if self.model is None:
            self.build_model(num_classes)
        
        # 定义回调函数
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=0.0001,
            verbose=1
        )
        
        print("开始训练模型...")
        
        # 训练模型（使用类别权重）
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            class_weight=self.class_weights,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # 评估模型
        print("\n测试集评估结果:")
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"损失: {loss:.4f}, 准确率: {accuracy:.4f}")
    
    def save_model(self, model_dir='sentiment_model_fixed'):
        """
        保存模型和相关组件
        """
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        self.model.save(os.path.join(model_dir, 'model.h5'))
        
        with open(os.path.join(model_dir, 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        with open(os.path.join(model_dir, 'label_mapping.pkl'), 'wb') as f:
            pickle.dump(self.label_mapping, f)
        
        print(f"模型已保存到 {model_dir} 目录")
    
    def load_model(self, model_dir='sentiment_model_fixed'):
        """
        加载模型和相关组件
        """
        self.model = tf.keras.models.load_model(os.path.join(model_dir, 'model.h5'))
        
        with open(os.path.join(model_dir, 'tokenizer.pkl'), 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        with open(os.path.join(model_dir, 'label_mapping.pkl'), 'rb') as f:
            self.label_mapping = pickle.load(f)
        
        print(f"模型已从 {model_dir} 目录加载")
    
    def predict(self, text):
        """
        预测文本的情感
        """
        if self.model is None or self.tokenizer is None or self.label_mapping is None:
            raise ValueError("请先加载或训练模型")
        
        # 预处理文本
        tokenized_text = self.tokenize_chinese(text)
        sequence = self.tokenizer.texts_to_sequences([tokenized_text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_len)
        
        # 预测
        pred_probs = self.model.predict(padded_sequence, verbose=0)[0]
        pred_class = np.argmax(pred_probs)
        
        # 转换为情感标签
        inverse_mapping = {v: k for k, v in self.label_mapping.items()}
        pred_label = inverse_mapping[pred_class]
        
        return pred_label, pred_probs[pred_class]

    def interactive_predict(self):
        """
        交互式预测功能 - 允许用户实时输入文本进行情感分析
        """
        if self.model is None or self.tokenizer is None or self.label_mapping is None:
            print("错误：请先加载或训练模型")
            return
        
        print("\n" + "="*60)
        print("交互式情感分析预测")
        print("="*60)
        print("输入文本进行情感分析（输入 'quit' 或 '退出' 结束）")
        print("支持的情感标签:", list(self.label_mapping.keys()))
        print("-"*60)
        
        while True:
            try:
                # 获取用户输入
                text = input("\n请输入文本: ").strip()
                
                # 检查退出条件
                if text.lower() in ['quit', '退出', 'exit', 'q']:
                    print("感谢使用交互式情感分析！")
                    break
                
                # 检查空输入
                if not text:
                    print("请输入有效的文本")
                    continue
                
                # 进行预测
                label, prob = self.predict(text)
                
                # 显示结果
                print(f"预测结果:")
                print(f"  文本: {text}")
                print(f"  情感: {label}")
                print(f"  置信度: {prob:.4f}")
                
                # 显示所有可能的概率分布
                tokenized_text = self.tokenize_chinese(text)
                sequence = self.tokenizer.texts_to_sequences([tokenized_text])
                padded_sequence = pad_sequences(sequence, maxlen=self.max_len)
                pred_probs = self.model.predict(padded_sequence, verbose=0)[0]
                
                print(f"  详细概率分布:")
                inverse_mapping = {v: k for k, v in self.label_mapping.items()}
                for i, prob_val in enumerate(pred_probs):
                    emotion = inverse_mapping[i]
                    print(f"    {emotion}: {prob_val:.4f}")
                
                print("-"*40)
                
            except KeyboardInterrupt:
                print("\n\n检测到中断，退出交互模式")
                break
            except Exception as e:
                print(f"预测过程中出现错误: {e}")
                print("请重新输入文本")

def test_fixed_model():
    """
    测试修复后的模型
    """
    # 数据集路径
    data_path = 'data/chinese_emotion_analysis_100k.csv'
    
    if not os.path.exists(data_path):
        print(f"错误：未找到数据集文件 {data_path}")
        return
    
    print("="*60)
    print("修复的情感分析模型测试")
    print("="*60)
    
    # 创建模型实例
    model = FixedSentimentAnalysisModel(
        max_words=5000,
        max_len=50,
        embedding_dim=100
    )
    
    # 加载和预处理数据
    print("加载并预处理数据...")
    X_train, X_test, y_train, y_test = model.load_data(data_path)
    
    # 训练模型
    print("开始训练模型...")
    model.train(X_train, y_train, X_test, y_test, epochs=15, batch_size=32)
    
    # 保存模型
    print("保存模型...")
    model.save_model()
    
    # 测试预测功能
    print("\n测试预测功能...")
    test_texts = [
        "今天天气真好，心情非常愉快！",
        "这件事让我很生气，太不公平了",
        "我不知道该怎么办，有点迷茫",
        "这部电影太精彩了，强烈推荐给大家",
        "吃了一口生虫子的面包，味道很油腻，特别反感。",
        "室友总是说八卦，真的很让人反感。",
        "吃到了想吃很久的烤肉，味道超棒，心情很开心。"
    ]
    
    for text in test_texts:
        label, prob = model.predict(text)
        print(f"文本: {text}")
        print(f"预测情感: {label}, 概率: {prob:.4f}\n")

if __name__ == "__main__":
    test_fixed_model()
