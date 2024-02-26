import torch
import torch.nn as nn
import torch.optim as optim

# 定义Sora模型
class SoraDiffusionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SoraDiffusionModel, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.diffusion_layer = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
        )
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x =self.fc1(x)
        x = self.diffusion_layer(x)
        x = self.fc2(x)
        return x

# 初始化模型
model = SoraDiffusionModel(input_size=10, hidden_size=50, output_size=2)

# 定义损失函数和优化器
criterion = nn.MSELoss()

# 使用均方误差作为损失函数
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
   
# 模拟数据
inputs = torch.randn(16, 10)
targets = torch.randn(16, 2)     
# 训练模型
for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 ==0:
        print(f'Epoch: [{epoch +1}/ 100], Loss: {loss.item():.4f}')

# 测试模型
with torch.no_grad():
    predictions = model(inputs)
    print(predictions)