import torch 
import torch.nn as nn
import torch.optim as optim

# 定义模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        x = self.fc(x)

        return x

# 初始化模型
model = SimpleModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# 定义损失函数
def lodd_fn(output,target,  human_feedback):
    loss = nn.MSELoss()(output, target) # 计算均方差损失
    loss += human_feedback * nn.MSELoss()(output, human_feedback) # 加入人类反馈损失
    
    return loss

# 模拟人类反馈数据
human_feedback = torch.randn(1,1)

# 训练模型
for epoch in range(1000):
    # 随机生成一批数据
    data = torch.randn(10, 10)
    target = torch.randn(10, 1)
    
    # 前向传播
    output = model(data)
    
    # 损失函数
    loss = lodd_fn(output=output, target=target, human_feedback=human_feedback)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch [{epoch + 1}/ 1000], Loss: {loss.item()}')
