import torch
import torch.nn as nn
import torch.optim as optim

class ImageClassifier(nn.Module):
    def __init__(self,num_classes):
        super(ImageClassifier, self).__init__()
        
        # 卷积层快
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # 全连接层用于分类
        self.fc_layers = nn.Sequential(
            nn.Linear(32 * 32, 64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64, num_classes),
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(-1, 32 * 32 * 32)
        x = self.fc_layers(x)
        return x

# 实例化模型
model = ImageClassifier(num_classes=4)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 查看模型摘要信息
print(model)
        