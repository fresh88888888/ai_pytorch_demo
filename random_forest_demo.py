import torch
import torch.nn as nn
import torch.optim as optim

# 定义随机森林模型
class RandomForest(nn.Module):
    def __init__(self):
        super(RandomForest, self).__init__()
        
        self.fc1 = nn.Linear(2, 2)  # 两个特征体重、颜色
        
    def forward(self, x):
        x = self.fc1(x)
        
        return x

# 训练数据
data = torch.tensor([[5.0,0], [4.0,1.0],[3.0,0], [6.0,1.0]], dtype=torch.float32)
labels = torch.tensor([0,1,0,1], dtype=torch.long)

# 模型、损失函数和优化器
model = RandomForest()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

#训练过程
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(data)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 ==0:
        print(f'Epoch: {epoch + 1}/100, Loss: {loss.item():.4f}')
    
# 测试模型 
with torch.no_grad():
    outputs = model(data)
    _, predicted = torch.max(outputs, 1)
    correct = (predicted==labels).sum().item()
    
    print(f'Accuracy: {correct / len(labels * 100)}%')
