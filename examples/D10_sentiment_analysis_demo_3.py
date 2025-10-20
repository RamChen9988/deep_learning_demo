# 导入必要的库
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from gensim.models import KeyedVectors

import jieba

import pickle
import os

# 设置随机种子，保证结果可复现
np.random.seed(42)
tf.random.set_seed(42)

class SentimentAnalysisModel:
    def __init__(self, max_words=10000, max_len=100, embedding_dim=128, use_pretrained_embeddings=False, embedding_path=None):
        """
        初始化情感分析模型参数
        
        参数:
            max_words: 最大词汇量
            max_len: 句子最大长度
            embedding_dim: 词嵌入维度
            use_pretrained_embeddings: 是否使用预训练词向量
            embedding_path: 预训练词向量文件路径
        """
        self.max_words = max_words
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.use_pretrained_embeddings = use_pretrained_embeddings
        self.embedding_path = embedding_path
        self.tokenizer = None
        self.label_mapping = None
        self.model = None
        self.embedding_matrix = None

    # 定义分词函数（在加载数据时调用）
    def tokenize_chinese(self, text):
    # 分词后用空格连接（方便Tokenizer处理）
        return ' '.join(jieba.cut(text))
    
    def load_pretrained_embeddings(self):
        """
        加载预训练词向量并构建嵌入矩阵
        """
        if not self.use_pretrained_embeddings or not self.embedding_path:
            print("未启用预训练词向量或未指定词向量文件路径")
            return
        
        print(f"正在加载预训练词向量: {self.embedding_path}")
        
        try:
            # 加载预训练词向量
            w2v_model = KeyedVectors.load_word2vec_format(self.embedding_path, binary=True)
            
            # 构建嵌入矩阵（适配Tokenizer的词汇表）
            self.embedding_matrix = np.zeros((self.max_words, self.embedding_dim))
            found_words = 0
            total_words = min(self.max_words, len(self.tokenizer.word_index))
            
            for word, i in self.tokenizer.word_index.items():
                if i < self.max_words:
                    if word in w2v_model:
                        self.embedding_matrix[i] = w2v_model[word]  # 用预训练向量初始化
                        found_words += 1
            
            coverage = found_words / total_words * 100
            print(f"预训练词向量覆盖情况: {found_words}/{total_words} ({coverage:.2f}%)")
            
        except Exception as e:
            print(f"加载预训练词向量失败: {e}")
            print("将使用随机初始化的嵌入层")
            self.embedding_matrix = None
    
    def load_data(self, file_path):
        """
        加载并预处理数据集
        
        参数:
            file_path: 数据集文件路径
            
        返回:
            预处理后的训练集和测试集
        """
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 检查必要的列是否存在
        if 'text' not in df.columns or 'label' not in df.columns:
            raise ValueError("数据集必须包含'text'和'label'列")
        
        # 提取文本和标签
        texts = df['text'].astype(str).tolist()
        texts = [self.tokenize_chinese(text) for text in texts]  # 中文分词
        labels = df['label'].tolist()
        
        # 创建标签映射（将文本标签转换为数字）
        unique_labels = list(set(labels))
        self.label_mapping = {label: i for i, label in enumerate(unique_labels)}
        inverse_mapping = {v: k for k, v in self.label_mapping.items()}
        
        # 打印标签信息
        print(f"发现的情感标签: {unique_labels}")
        print(f"标签映射: {self.label_mapping}")
        
        # 将标签转换为数字
        labels = [self.label_mapping[label] for label in labels]
        
        # 初始化Tokenizer
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.tokenizer.fit_on_texts(texts)
        
        # 如果启用预训练词向量，加载词向量
        if self.use_pretrained_embeddings:
            self.load_pretrained_embeddings()
        
        # 将文本转换为序列
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # 对序列进行填充，使它们具有相同的长度
        X = pad_sequences(sequences, maxlen=self.max_len)
        
        # 将标签转换为独热编码
        y = to_categorical(labels)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, num_classes):
        """
        构建LSTM情感分析模型
        
        参数:
            num_classes: 情感类别数量
        """
        # 构建嵌入层
        if self.use_pretrained_embeddings and self.embedding_matrix is not None:
            print("使用预训练词向量初始化嵌入层")
            embedding_layer = Embedding(
                input_dim=self.max_words,
                output_dim=self.embedding_dim,
                input_length=self.max_len,
                weights=[self.embedding_matrix],  # 加载预训练权重
                trainable=False  # 初期冻结，避免破坏预训练知识
            )
        else:
            print("使用随机初始化的嵌入层")
            embedding_layer = Embedding(
                input_dim=self.max_words,
                output_dim=self.embedding_dim,
                input_length=self.max_len
            )
        
        self.model = Sequential([
            # 嵌入层：将词汇索引转换为密集向量
            embedding_layer,
            # LSTM层：捕获序列信息
            LSTM(units=128, return_sequences=False),  # 设置为False只返回最后一个时间步的输出
            # Dropout层：防止过拟合
            Dropout(0.6),
            # 输出层：使用softmax激活函数进行多分类
            Dense(units=num_classes, activation='softmax')
        ])
        
        # 编译模型
        # 调整学习率（如用Adam(learning_rate=0.0001)，避免学习过快）
        """ self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        ) """

        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # 打印模型结构
        self.model.summary()
    
    def train(self, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
        """
        训练模型
        
        参数:
            X_train, y_train: 训练数据和标签
            X_test, y_test: 测试数据和标签
            epochs: 训练轮数
            batch_size: 批次大小
        """
        # 计算类别数量
        num_classes = y_train.shape[1]
        
        # 如果模型未构建，则构建模型
        if self.model is None:
            self.build_model(num_classes)
        
        # 定义早停回调函数，防止过拟合
        early_stopping = EarlyStopping(
         monitor='val_loss',  # 监控验证集损失
         patience=3,  # 3轮无提升则停止
         restore_best_weights=True  # 恢复最优权重
        )
        # 加入早停机制（Early Stopping）
        
        # model.fit(..., callbacks=[early_stopping])
        
        # 训练模型
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            validation_data=(X_test, y_test)
        )
        
        # 评估模型
        print("\n测试集评估结果:")
        loss, accuracy = self.model.evaluate(X_test, y_test)
        print(f"损失: {loss:.4f}, 准确率: {accuracy:.4f}")
    
    def save_model(self, model_dir='sentiment_model'):
        """
        保存模型和相关组件
        
        参数:
            model_dir: 保存模型的目录
        """
        # 创建目录（如果不存在）
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 保存模型
        self.model.save(os.path.join(model_dir, 'model.h5'))
        
        # 保存tokenizer
        with open(os.path.join(model_dir, 'tokenizer.pkl'), 'wb') as f:
            pickle.dump(self.tokenizer, f)
        
        # 保存标签映射
        with open(os.path.join(model_dir, 'label_mapping.pkl'), 'wb') as f:
            pickle.dump(self.label_mapping, f)
        
        print(f"模型已保存到 {model_dir} 目录")
    
    def load_model(self, model_dir='sentiment_model'):
        """
        加载模型和相关组件
        
        参数:
            model_dir: 模型所在目录
        """
        # 加载模型
        self.model = tf.keras.models.load_model(os.path.join(model_dir, 'model.h5'))
        
        # 加载tokenizer
        with open(os.path.join(model_dir, 'tokenizer.pkl'), 'rb') as f:
            self.tokenizer = pickle.load(f)
        
        # 加载标签映射
        with open(os.path.join(model_dir, 'label_mapping.pkl'), 'rb') as f:
            self.label_mapping = pickle.load(f)
        
        print(f"模型已从 {model_dir} 目录加载")
    
    def predict(self, text):
        """
        预测文本的情感
        
        参数:
            text: 要预测的文本
            
        返回:
            预测的情感标签和概率
        """
        # 检查模型是否已加载
        if self.model is None or self.tokenizer is None or self.label_mapping is None:
            raise ValueError("请先加载或训练模型")
        
        # 预处理文本
        sequence = self.tokenizer.texts_to_sequences([text])
        padded_sequence = pad_sequences(sequence, maxlen=self.max_len)
        
        # 预测
        pred_probs = self.model.predict(padded_sequence)[0]
        pred_class = np.argmax(pred_probs)
        
        # 转换为情感标签
        inverse_mapping = {v: k for k, v in self.label_mapping.items()}
        pred_label = inverse_mapping[pred_class]
        
        return pred_label, pred_probs[pred_class]

# 交互式预测功能
def interactive_prediction():
    """
    交互式预测功能：让用户输入文本进行情感预测，直到输入'quit'退出
    """
    # 创建模型实例并加载已训练的模型
    model = SentimentAnalysisModel()
    
    # 检查模型是否存在
    model_dir = 'sentiment_model'
    if not os.path.exists(model_dir):
        print(f"错误：未找到已训练的模型目录 {model_dir}")
        print("请先运行主函数训练模型")
        return
    
    try:
        # 加载模型
        print("加载情感分析模型...")
        model.load_model(model_dir)
        print("模型加载成功！")
    except Exception as e:
        print(f"加载模型失败: {e}")
        print("请确保模型已正确训练并保存")
        return
    
    print("\n" + "="*50)
    print("情感分析交互式预测")
    print("="*50)
    print("请输入文本进行情感分析，输入 'quit' 退出程序")
    print("="*50)
    
    while True:
        # 获取用户输入
        user_input = input("\n请输入文本: ").strip()
        
        # 检查是否退出
        if user_input.lower() == 'quit':
            print("感谢使用情感分析系统，再见！")
            break
        
        # 检查输入是否为空
        if not user_input:
            print("输入不能为空，请重新输入")
            continue
        
        try:
            # 进行预测
            label, prob = model.predict(user_input)
            
            # 显示预测结果
            print(f"\n预测结果:")
            print(f"文本: {user_input}")
            print(f"情感: {label}")
            print(f"置信度: {prob:.4f}")
            
        except Exception as e:
            print(f"预测过程中出现错误: {e}")
            print("请尝试输入其他文本")

# 演示使用预训练词向量的函数
def demo_with_pretrained_embeddings():
    """
    演示使用预训练词向量的情感分析模型
    """
    # 数据集路径
    data_path = 'data/chinese_emotion_analysis_100k.csv'
    
    # 检查数据集是否存在
    if not os.path.exists(data_path):
        print(f"错误：未找到数据集文件 {data_path}")
        print("请确保数据集文件存在于指定路径")
        return
    
    # 预训练词向量路径（需要用户自行下载）
    embedding_path = 'path/to/tencent-ailab-embedding-zh-d100-v0.2.0-s.bin'
    
    print("="*60)
    print("预训练词向量情感分析演示")
    print("="*60)
    
    # 检查预训练词向量文件是否存在
    if not os.path.exists(embedding_path):
        print(f"警告：未找到预训练词向量文件 {embedding_path}")
        print("请从 https://ai.tencent.com/ailab/nlp/en/download.html 下载")
        print("将使用随机初始化的嵌入层进行演示")
        use_pretrained = False
    else:
        use_pretrained = True
    
    # 创建模型实例（使用预训练词向量）
    model = SentimentAnalysisModel(
        max_words=10000,
        max_len=100,
        embedding_dim=100,  # 腾讯词向量维度为100
        use_pretrained_embeddings=use_pretrained,
        embedding_path=embedding_path if use_pretrained else None
    )
    
    # 加载和预处理数据
    print("\n加载并预处理数据...")
    X_train, X_test, y_train, y_test = model.load_data(data_path)
    
    # 训练模型
    print("\n开始训练模型...")
    model.train(X_train, y_train, X_test, y_test, epochs=5, batch_size=32)
    
    # 保存模型
    print("\n保存模型...")
    model.save_model('sentiment_model_pretrained')
    
    # 测试预测功能
    print("\n测试预测功能...")
    test_texts = [
        "今天天气真好，心情非常愉快！",
        "这件事让我很生气，太不公平了",
        "我不知道该怎么办，有点迷茫",
        "这部电影太精彩了，强烈推荐给大家"
    ]
    
    for text in test_texts:
        label, prob = model.predict(text)
        print(f"文本: {text}")
        print(f"预测情感: {label}, 概率: {prob:.4f}\n")

# 主函数：演示模型的训练、保存和调用过程
def main():
    # 数据集路径
    data_path = 'data/chinese_emotion_analysis_100k.csv'
    
    # 检查数据集是否存在
    if not os.path.exists(data_path):
        print(f"错误：未找到数据集文件 {data_path}")
        print("请确保数据集文件存在于指定路径")
        return
    
    print("="*60)
    print("情感分析模型演示")
    print("="*60)
    print("请选择运行模式:")
    print("1. 使用随机初始化嵌入层（默认）")
    print("2. 使用预训练词向量（需要下载词向量文件）")
    
    try:
        choice = input("请输入选择 (1 或 2): ").strip()
        if choice == "2":
            demo_with_pretrained_embeddings()
        else:
            # 创建模型实例（随机初始化）
            model = SentimentAnalysisModel()
            
            # 加载和预处理数据
            print("\n加载并预处理数据...")
            X_train, X_test, y_train, y_test = model.load_data(data_path)
            
            # 训练模型
            print("\n开始训练模型...")
            model.train(X_train, y_train, X_test, y_test, epochs=5, batch_size=32)
            
            # 保存模型
            print("\n保存模型...")
            model.save_model()
            
            # 加载模型（模拟新的会话）
            print("\n加载模型...")
            new_model = SentimentAnalysisModel()
            new_model.load_model()
            
            # 测试预测功能
            print("\n测试预测功能...")
            test_texts = [
                "今天天气真好，心情非常愉快！",
                "这件事让我很生气，太不公平了",
                "我不知道该怎么办，有点迷茫",
                "这部电影太精彩了，强烈推荐给大家"
            ]
            
            for text in test_texts:
                label, prob = new_model.predict(text)
                print(f"文本: {text}")
                print(f"预测情感: {label}, 概率: {prob:.4f}\n")
            
            # 启动交互式预测
            print("\n启动交互式预测功能...")
            interactive_prediction()
    
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"程序运行出错: {e}")

# 如果直接运行脚本，则执行主函数
if __name__ == "__main__":
    main()
