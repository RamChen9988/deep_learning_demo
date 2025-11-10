# 猫狗分类预测器

这是一个简单的深度学习模型预测程序，用于演示如何加载和使用训练好的迁移学习模型进行猫狗分类。

## 文件说明

- `simple_predict.py` - 交互式预测程序（需要手动输入图片路径）
- `simple_predict_direct.py` - 命令行预测程序（直接在命令行指定图片路径）

## 使用方法

### 方法1：命令行方式（推荐）
```bash
# 预测单张图片
python simple_predict_direct.py 图片路径

# 示例
python simple_predict_direct.py test_cat.jpg
python simple_predict_direct.py test_dog.jpg
```

### 方法2：交互式方式
```bash
python simple_predict.py
```

然后按照提示输入图片路径。

## 模型信息

- 模型文件：`../save_model/simple_transfer_learning_resnet18.pth`
- 模型架构：基于ResNet18的迁移学习模型
- 类别：猫(cat)和狗(dog)
- 输入尺寸：224x224像素
- 预处理：与ImageNet相同的标准化参数

## 依赖库

- torch
- torchvision
- Pillow (PIL)

## 注意事项

1. 确保模型文件 `simple_transfer_learning_resnet18.pth` 存在
2. 图片格式支持常见的图像格式（JPG、PNG等）
3. 程序会自动检测并使用GPU（如果可用）
