import torch
import torchvision
import pandas as pd
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from torch import nn, optim

# 定义数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# 定义图像数据集
image_data = datasets.ImageFolder('/path/to/image_data', transform=transform)
image_loader = DataLoader(image_data, batch_size=64, shuffle=True)

# 定义文本数据集（这里假设文本数据已经是一个Pandas DataFrame, 列名为'text'和'label'）
text_data= pd.read_csv('/path/to/text_data.csv')
text_loader = DataLoader(text_data, batch_size=64, shuffle=True)

# 定义一个多模态模型


class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()
        
        # 定义文本模型
        self.tex_model = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
        
        # 定义图像模型
        self.image_model = nn.Sequential(
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, text_data, image_data):
        # 文本模型前向传播
        text_output = self.tex_model(text_data)
        
        # 图像模型前向传播
        image_output = self.image_model(image_data)
        
        # 输出拼接
        output = torch.cat((text_data, image_data), 1)
        
        return output
    
# 实例化模型、损失函数和优化器
model = MultimodalModel()
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(10):
    for text_batch, image_batch, labels in zip(text_loader, image_loader , labels):
        # 前向传播
        outputs = model(text_batch, image_batch)
        
        # 计算损失
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        print(f'Epoch[{epoch + 1}/10], Loss: {loss.item():.4f}')
        