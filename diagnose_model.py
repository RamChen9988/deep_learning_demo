import pickle
import tensorflow as tf
import numpy as np
import os

def diagnose_model():
    """诊断模型问题"""
    model_dir = 'sentiment_model'
    
    # 检查模型文件是否存在
    if not os.path.exists(model_dir):
        print(f"错误：未找到模型目录 {model_dir}")
        return
    
    try:
        # 加载标签映射
        with open(os.path.join(model_dir, 'label_mapping.pkl'), 'rb') as f:
            label_mapping = pickle.load(f)
        print(f"标签映射: {label_mapping}")
        
        # 加载tokenizer
        with open(os.path.join(model_dir, 'tokenizer.pkl'), 'rb') as f:
            tokenizer = pickle.load(f)
        print(f"词汇表大小: {len(tokenizer.word_index)}")
        
        # 加载模型
        model = tf.keras.models.load_model(os.path.join(model_dir, 'model.h5'))
        print("模型加载成功")
        
        # 检查模型结构
        print("\n模型结构:")
        model.summary()
        
        # 检查模型权重
        print("\n检查模型权重:")
        for layer in model.layers:
            print(f"层: {layer.name}")
            if layer.get_weights():
                weights = layer.get_weights()[0]
                print(f"  权重形状: {weights.shape}")
                print(f"  权重均值: {np.mean(weights):.6f}")
                print(f"  权重标准差: {np.std(weights):.6f}")
                print(f"  权重范围: [{np.min(weights):.6f}, {np.max(weights):.6f}]")
        
        # 测试预测
        print("\n测试预测:")
        test_texts = [
            "今天天气真好，心情非常愉快！",
            "这件事让我很生气，太不公平了",
            "我不知道该怎么办，有点迷茫",
            "这部电影太精彩了，强烈推荐给大家"
        ]
        
        for text in test_texts:
            sequence = tokenizer.texts_to_sequences([text])
            padded_sequence = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=100)
            pred_probs = model.predict(padded_sequence)[0]
            pred_class = np.argmax(pred_probs)
            
            inverse_mapping = {v: k for k, v in label_mapping.items()}
            pred_label = inverse_mapping[pred_class]
            
            print(f"文本: {text}")
            print(f"预测概率分布: {pred_probs}")
            print(f"预测情感: {pred_label}, 概率: {pred_probs[pred_class]:.4f}")
            print(f"预测类别: {pred_class}")
            print("-" * 50)
            
    except Exception as e:
        print(f"诊断过程中出错: {e}")

if __name__ == "__main__":
    diagnose_model()
