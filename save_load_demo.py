import torch
import torch.nn as nn
import torch.optim as optim


# 定义一个简单的全连接神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forwar(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 模型实例化
model = SimpleNet()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 模拟训练过程，此处省略
# ...

# 保存模型参数
torch.save(model.state_dict(), 'model_weights.pth')

# 加载模型参数(假设在另一个设备上运行)
model = SimpleNet()   # 创新创建模型结构，因为Pytorch的模型是不可序列化的，因此需要重新创建模型结构实例

m_obj = torch.load('model_weights.pth')
print(m_obj)
model.load_state_dict(m_obj)  # 加载模型参数到模型实例中

