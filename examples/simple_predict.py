"""
简单的模型预测程序
加载训练好的迁移学习模型进行猫狗分类预测
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import os

class SimplePredictor:
    def __init__(self, model_path='../save_model/simple_transfer_learning_resnet18.pth'):
        """
        初始化预测器
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 定义数据预处理（与训练时一致）
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
        # 类别名称（猫和狗）
        self.class_names = ['cat', 'dog']
        
        # 加载模型
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        """
        加载训练好的模型
        """
        print(f"加载模型: {model_path}")
        
        # 创建与训练时相同的模型结构
        model = models.resnet18(weights=None)  # 不使用预训练权重
        num_features = model.fc.in_features
        model.fc = nn.Linear(num_features, 2)  # 2个类别：猫和狗
        
        # 加载训练好的权重
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model = model.to(self.device)
        model.eval()  # 设置为评估模式
        
        print("✅ 模型加载成功")
        return model
    
    def predict_image(self, image_path):
        """
        预测单张图片
        """
        # 检查图片是否存在
        if not os.path.exists(image_path):
            print(f"❌ 图片不存在: {image_path}")
            return None
        
        try:
            # 加载并预处理图片
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0)  # 添加batch维度
            input_tensor = input_tensor.to(self.device)
            
            # 预测
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                predicted_class = torch.argmax(outputs, 1).item()
                confidence = probabilities[0][predicted_class].item()
            
            # 返回结果
            result = {
                'predicted_class': self.class_names[predicted_class],
                'confidence': confidence,
                'class_index': predicted_class
            }
            
            print(f"📷 图片: {os.path.basename(image_path)}")
            print(f"🔮 预测结果: {result['predicted_class']}")
            print(f"📊 置信度: {result['confidence']:.4f}")
            
            return result
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
            return None

def main():
    """
    主函数 - 简单的交互式预测
    """
    print("=" * 50)
    print("🐱🐶 猫狗分类预测器")
    print("=" * 50)
    
    # 初始化预测器
    predictor = SimplePredictor()
    
    while True:
        print("\n请输入图片路径 (输入 'quit' 退出):")
        image_path = input().strip()
        
        if image_path.lower() == 'quit':
            print("👋 再见！")
            break
        
        # 执行预测
        result = predictor.predict_image(image_path)
        
        if result:
            print(f"\n🎯 最终预测: {result['predicted_class']} (置信度: {result['confidence']:.2%})")

if __name__ == "__main__":
    main()
