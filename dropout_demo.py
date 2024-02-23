import torch
import torch.nn as nn

# 假设我们构建一个多层感知机，包含两个隐藏层和一个输出层
class SimpleMLP(nn.Module):
    def __init__(self, input_size = 784, hidden_size = 256, output_size = 10, dropout_prob = 0.5):
        super(SimpleMLP, self).__init__()
        
        # 隐藏层的定义
        self.hidden_layer1 = nn.Linear(input_size, hidden_size)
        # 第一个隐藏层后的Dropout层
        self.dropout1 = nn.Dropout(dropout_prob)
        
        self.hidden_layer2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_prob)
        
        self.output_layer = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # 输入数据经过第一个隐藏层并激活
        x = torch.relu(self.hidden_layer1(x))
        # 应用Dropout，按概率dropout_prob丢弃一些神经元
        x = self.dropout1(x)
        
        # 继续通过第二隐藏层并激活
        x = torch.relu(self.hidden_layer2(x))
        # 应用Dropout，按概率dropout_prob丢弃一些神经元
        x = self.dropout2(x)
        
        # 最后通过输出层得到预测结果
        x = self.output_layer(x)

        return x

# 创建模型实例
model = SimpleMLP()

# 假设我们有一些输入数据
inputs = torch.randn(100, 784) # 100个样本，每个样本784个特征

# 前向传播过程，dropout会砸训练模式下生效
outputs = model(inputs)

# 在训练过程中，Dropout层会根据设定的概率进行丢弃
# 而在验证和测试阶段，通常会关闭Dropout以保持模型行为的一致性
model.train() # 设置模型为训练模式
with torch.set_grad_enabled(True):
    # 确保梯度计算开启
    outputs_with_dropout = model(inputs)  # 这里的Dropout会起作用
    print(outputs_with_dropout)

# 当需要验证和测试时，一般会禁用Dropout
model.eval() # 设置模型为评估模式
with torch.no_grad():  # 不进行梯度计算
    outputs_with_dropout = model(inputs)  # 在评估时Dropout不会生效
    print(outputs_with_dropout)

