from tensorflow.keras.layers import Bidirectional, Attention, GlobalMaxPool1D

self.model = Sequential([
    Embedding(...),  # 用预训练嵌入层
    Bidirectional(LSTM(128, return_sequences=True)),  # 双向LSTM捕获上下文
    GlobalMaxPool1D(),  # 提取关键特征
    Dense(64, activation='relu'),  # 增加全连接层
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

#已经在10_sentiment_analysis_demo_2.py 中加入，请参考该文件。