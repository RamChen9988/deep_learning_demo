#分析数据分布
#先检查数据集的类别平衡和样本量
#如果 “平静” 占比超过 60%，需要处理类别不平衡。
#如果总样本量小于 5000，需要考虑补充数据或简化任务

import pandas as pd

# 加载数据集
df = pd.read_csv('data/Simplified_Chinese_Multi-Emotion_Dialogue_Dataset.csv')

# 打印类别分布
print("类别分布：")
print(df['label'].value_counts(normalize=True))  # 查看每个类别的占比
print("\n总样本量：", len(df))

# 如果 “平静” 占比超过 60%，需要处理类别不平衡。
# 如果总样本量小于 5000，需要考虑补充数据或简化任务。
# 处理类别不平衡
# 方法 1：下采样多数类：随机减少 “平静” 类样本，使各类别比例接近（如最大占比不超过 50%）。
# 方法 2：上采样少数类：复制少数类样本（或用数据增强生成相似文本，如同义词替换）。
# 方法 3：训练时加权：在模型编译时给少数类更高的权重，
# from sklearn.utils.class_weight import compute_class_weight

# # 计算类别权重（平衡少数类）
# y_train_labels = np.argmax(y_train, axis=1)  # 将独热编码转回数字标签
# class_weights = compute_class_weight('balanced', classes=np.unique(y_train_labels), y=y_train_labels)
# class_weight_dict = {i: w for i, w in enumerate(class_weights)}

# # 训练时传入权重
# model.fit(..., class_weight=class_weight_dict)