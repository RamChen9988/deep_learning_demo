# 假设已下载中文预训练Word2Vec（如腾讯AI Lab的词向量）
from gensim.models import KeyedVectors

# 加载预训练词向量（需自行下载，如https://ai.tencent.com/ailab/nlp/en/download.html）
w2v_model = KeyedVectors.load_word2vec_format('path/to/tencent-ailab-embedding-zh-d100-v0.2.0-s.bin', binary=True)

# 构建嵌入矩阵（适配Tokenizer的词汇表）
embedding_matrix = np.zeros((self.max_words, self.embedding_dim))
for word, i in self.tokenizer.word_index.items():
    if i < self.max_words and word in w2v_model:
        embedding_matrix[i] = w2v_model[word]  # 用预训练向量初始化

# 在模型中使用预训练嵌入层（冻结或微调）
Embedding(
    input_dim=self.max_words,
    output_dim=self.embedding_dim,
    input_length=self.max_len,
    weights=[embedding_matrix],  # 加载预训练权重
    trainable=False  # 初期冻结，避免破坏预训练知识
)