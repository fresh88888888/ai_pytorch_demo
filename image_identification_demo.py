import torch
from torchvision import transforms, models, io

# 加载预训练模型ResNet-50
model = models.resnet50(weights=True)
model = model.eval()

# 加载并预处理图像
transform = transforms.Compose([
                # 将 numpy 数组转换为PIL Image对象
                transforms.ToPILImage(),
                transforms.Resize(256),
                # 根据模型需求调整大小
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

# 加载图片
image_path = '18771871.jpg'
img = io.read_image(image_path)

# 应用转换器
img_tensor = transform(img)

# 将图片Tensor放入一个Batch中，对于大多数模型输入数据需要是批次形式
img_tensor = img_tensor.unsqueeze(0)

# 将模型输出转化为概率分布
with torch.no_grad():
    outputs = model(img_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

# 获取最大概率的类别索引
_, predicted_idx = torch.max(outputs, 1)

# 输出类别索引
print(predicted_idx.item())

