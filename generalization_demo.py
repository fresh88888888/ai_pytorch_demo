import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 假设我们有一些模拟的数据点，训练集和验证集各一半
x_train = torch.randn(1000, 1)
y_train = 2 * x_train + 1 + torch.randn(1000, 1)  # 训练数据目标值，模拟线性关系并增加噪声

# 划分训练集和验证集
x_val = x_train[:500]
y_val = y_train[:500]
x_train = x_train[:500]
y_train = y_train[:500]

# 创建TensorDataset并将数据转化为DataLoader以便批量处理
train_data = TensorDataset(x_train, y_train)
val_data = TensorDataset(x_val, y_val)
train_loader = DataLoader(train_data, batch_size=32)
val_loader = DataLoader(val_data, batch_size=32)

# 定义一个简单的线性回归模型
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()

        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

# 模型实例化
model = LinearModel()

# 使用均方误差损失函数和SGD优化器
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 循环训练，这里简化未单个epoch
for epoch in range(10):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        # 正向传递计算预测值
        prediction = model(inputs)
        loss = criterion(prediction, targets)
        loss.backward()
        optimizer.step()
        
    with torch.no_grad():
        val_loss = 0.0
        for inputs, targets in val_loader:
            preds = model(inputs)
            val_loss = criterion(preds, targets).item() * len(inputs)
            
        val_loss /= len(val_data)
        
    print(f'Epoch {epoch + 1}, validation Loss: {val_loss:.4f}')