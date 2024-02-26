import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 定义一个简单的线性回归模型
class LineRegressionModel(nn.Module):
    def __init__(self, input_size, ouput_size):
        super(LineRegressionModel, self).__init__()
        
        self.linear = nn.Linear(input_size, ouput_size)
        
    def forward(slef, x):
        out = slef.linear(x)
        
        return out
    
# 定义损失函数和优化器
model = LineRegressionModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.get_parameter(), lr=0.01)

# 训练模型
num_epochs = 10
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # 假设我们使用一个数据加载器来获取训练数据和标签
        inputs = inputs.view(-1, input_size)
        # 假设我们的输入大小为input_size
        labels = labels.view(-1)
        # 假设我们的标签大小为output_size
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # 反向传播和泛化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i + 1) % 100 == 0:
            # 每100个batch打印一次损失值
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{len(i + 1)}/{train_loader}], Loss: {loss.item()}')
            