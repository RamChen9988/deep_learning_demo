#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
交互式情感分析演示
"""

import os
import sys
from D10_sentiment_analysis_demo_4 import FixedSentimentAnalysisModel

def demo_interactive_predict():
    """
    演示交互式预测功能
    """
    print("="*60)
    print("交互式情感分析演示")
    print("="*60)
    
    # 创建模型实例
    model = FixedSentimentAnalysisModel()
    
    # 检查模型是否已存在
    model_dir = 'sentiment_model_fixed'
    if os.path.exists(os.path.join(model_dir, 'model.h5')):
        print("检测到已训练的模型，正在加载...")
        model.load_model(model_dir)
        print("模型加载成功！")
    else:
        print("未找到已训练的模型，请先运行训练脚本")
        print("运行: python fixed_sentiment_analysis.py")
        return
    
    # 启动交互式预测
    model.interactive_predict()

def quick_test():
    """
    快速测试几个示例
    """
    print("\n" + "="*60)
    print("快速测试示例")
    print("="*60)
    
    model = FixedSentimentAnalysisModel()
    model_dir = 'sentiment_model_fixed'
    
    if os.path.exists(os.path.join(model_dir, 'model.h5')):
        model.load_model(model_dir)
        
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
            print(f"预测情感: {label}, 置信度: {prob:.4f}")
            print("-" * 40)
    else:
        print("未找到已训练的模型，请先运行训练脚本")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_test()
    else:
        demo_interactive_predict()
