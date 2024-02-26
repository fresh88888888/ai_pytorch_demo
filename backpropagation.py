import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的神经网络
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        
        self.layer1 = nn.Linear(in_features=10, out_features=5)
        self.layer2 = nn.Linear(in_features=5, out_features=2)
        
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.layer2(x)
         
        return x

# 创建一个模型实例
model = SimpleNN()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 生成一些随机数据作为输入和目标
inputs = torch.randn(64, 10)
targets = torch.randint(0,2, (64,))

# 前向传播
ouputs = model(inputs)
loss = criterion(ouputs, targets)

# 反向传播
optimizer.zero_grad()  # 清楚之前的梯度
loss.backward()        # 反向传播计算梯度
optimizer.step()       # 更新模型参数

print('Loss after backward propagation: ', loss.item())
