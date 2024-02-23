import torch
import torch.nn as nn

# 定义Sequential
model = nn.Sequential(
    # 输入层和隐藏层的线性层
    nn.Linear(10, 20),
    # 激活函数
    nn.ReLU(),
    
    # 隐藏层和输出层的线性层
    nn.Linear(20, 10),
    nn.ReLU(),
)

# 输入数据
input_data = torch.randn(1, 10)

# 前向传播
output_data = model(input_data)

# 打印输出数据
print(output_data)
