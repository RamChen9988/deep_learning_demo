from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(
    monitor='val_loss',  # 监控验证集损失
    patience=3,  # 3轮无提升则停止
    restore_best_weights=True  # 恢复最优权重
)
model.fit(..., callbacks=[early_stopping])

#已经在10_sentiment_analysis_demo_2.py 中加入，请参考该文件。