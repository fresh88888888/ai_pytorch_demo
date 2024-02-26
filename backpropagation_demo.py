import torch
import torch.nn as nn
import torch.optim as optim

# 定义一个简单的神经网络


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.fc1 = nn.Linear(1,1)

    def forward(self, x):
        x = self.fc1(x)

        return x


# 创建一个模型实例
model = Net()

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

# 训练数据
x_train = torch.tensor([[1.0], [2.0], [3.0]], requires_grad=True)
y_train = torch.tensor([[2.0], [4.0], [6.0]])

# 训练过程
for epoch in range(10):
    # 前向传播
    outputs = model(x_train)
    loss = criterion(outputs, y_train)
    
    # 反向传播和优化
    optimizer.zero_grad() # 清空过往梯度
    loss.backward()       # 反向传播，计算当前梯度
    optimizer.step()      # 更新参数
    
    print(f'Epoch: {epoch + 1}, Loss: {loss.item():.4f}')
