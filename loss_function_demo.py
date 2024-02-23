import torch
import torch.nn as nn


# 定义一个简单的线性模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        
        self.linear = nn.Linear(1, 1)
        
    def forward(self, x):
        return self.linear(x)
    
# 初始化模型，优化器以及损失函数
model = SimpleModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 假设我们有一些输入数据和对应的标签
inputs = torch.tensor([[1.0],[2.0], [3.0]], dtype=torch.float32)
targets = torch.tensor([[2.0], [4.0], [6.0]], dtype=torch.float32)

# 将数据送入模型并计算预测值
predictions = model(inputs)

# 计算损失
loss = loss_fn(predictions, targets)  #MSE损失是预测值和目标值之间的平方差的平均

# 打印当前损失
print('Current loss: {loss.item():.4f}', loss.item())

# 反向传播即参数更新
optimizer.zero_grad()  # 清零梯度缓冲区
loss.backward()        # 计算梯度
optimizer.step()       # 根据梯度更新模型参数



