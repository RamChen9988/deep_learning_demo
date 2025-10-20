import jieba

# 定义分词函数（在加载数据时调用）
def tokenize_chinese(text):
    # 分词后用空格连接（方便Tokenizer处理）
    return ' '.join(jieba.cut(text))

# 在load_data函数中，对texts进行分词：
texts = [tokenize_chinese(text) for text in df['text'].astype(str).tolist()]


# 这样可以更好地处理中文文本，提高模型效果。
# 注意：需要先安装jieba库（pip install jieba）
# 另外，确保在Tokenizer初始化时设置合适的参数，如oov_token='<OOV>'以处理未登录词。
# tokenizer = Tokenizer(num_words=self.max_words, oov_token='<OOV>')
# 然后继续后续的文本转换和填充步骤。
# 在10_sentiment_analysis_demo_2.py 中加入，请参考该文件。