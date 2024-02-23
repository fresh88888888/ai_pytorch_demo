import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 假设我们有一些随机数据来模拟训练过程，这里我们创建一些随机数据作为示例
features = torch.randn(100, 10) # 100个样本，每个样本10个特征
targets = torch.randint(0, 2, (100,))  # 100哥目标值，假设是二分类问题

# 将数据转换为PyTorch数据集
dataset = TensorDataset(features, targets)

# 创建数据加载器
train_loader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)

# 定义一个简单模型
class SimpleModel(nn.Module):
    def __init(self):
        super(SimpleModel, self).__init__()
        self.layer = nn.Linear(10, 2)  # 输入维度是10， 输出维度是2
        
    def forward(self, x):
        return self.layer(x)
    
# 实例化模型、损失函数和优化器
model = SimpleModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

#训练模型
num_epochs = 3  # 训练3个epoch
for epoch in range(num_epochs):
    # 训练阶段
    model.train()  # 设置模型为训练模式
    train_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()       # 清除之前的梯度
        outputs = model(inputs)     # 前向传播
        loss = criterion(outputs, targets)   # 计算损失
        loss.backward()                      # 反向传播
        optimizer.step()                     # 更新权重
        train_loss += loss.item() * inputs.size(0) # 累加损失
    
    # 计算平均训练损失
    train_loss /= len(train_loader.dataset)
    print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}')
    
    # 验证阶段（这里我们使用训练数据作为验证数据，实际中应该使用不同的数据集）
    model.eval()  # 设置模型为评估模式
    valid_loss = 0.0
    with torch.no_grad():  # 在验证阶段不计算梯度
        for inputs, targets in train_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            valid_loss += loss.item() * inputs.size(0)
            
    # 计算平均验证损失
    valid_loss /= len(train_loader.dataset)
    print(f'Epoch {epoch + 1} Validation Loss: {valid_loss:.4f}')
    

