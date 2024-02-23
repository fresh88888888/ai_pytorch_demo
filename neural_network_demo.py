import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 定义一个简单的两层全连接神经网络模块
class TwoLayoutNet(nn.Module):
    def __init__(self,input_size, hidden_size, num_classes):
        super(TwoLayoutNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
# 创建一个随机的小数据集, 假设我们有100个样本，每个样本5个特征
X = torch.randn(100, 5)
# 假设这是二分类问题，标签为0或1
y = torch.randint(0, 2, (100,))

#将数据转换为TensorDataSet并创建DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 初始化模型
model= TwoLayoutNet(36, 24, 100)

print('Training finshied.')
