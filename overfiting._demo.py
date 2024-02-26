import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 生成模拟数据
x_train =np.random.rand(100, 1)
y_train = 2 * x_train + 3 + np.random.randn(100, 1) * 0.3
x_test = np.random.rand(20, 1)
y_test = 2 * x_test + 3 + np.random.randn(20, 1) * 0.3

# 定义模型
class LinearRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegression, self).__init__()

        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.linear(x)
        
        return x

model = LinearRegression(1, 1)
# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 训练模型
for epoch in range(1000):
    # 前向传播
    outputs = model(torch.from_numpy(x_train).float())
    loss = criterion(outputs, torch.from_numpy(y_train).float())
    
    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f'Epoch [{epoch + 1}/ 1000], Loss: {loss.item()}')
        
        
# 绘制结果
plt.scatter(x_train.flatten(), y_train.flatten(), c='orange')
plt.plot(x_train.flatten(), model(
    torch.from_numpy(x_train).float()).detach().numpy(), 'g-', lw=1)
plt.show()
