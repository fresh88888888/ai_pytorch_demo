import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets


# 定义超参数
input_size = 10        # 输入图像的维度
output_size = 10
sequence_length = 10
hidden_size = 20       # 隐藏层神经元数量
num_layers = 1         # RNN层数
num_classes = 5        # 输出类别的数量（0~9）
num_epochs = 100       # 训练轮数
batch_sise = 10        # 批处理大小
learning_rate = 0.01   # 学习率

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # 使用RNN层
        self.rnn = nn.RNN(input_size,  hidden_size, num_layers, batch_first=True)
        # 全连接层
        self.fc = nn.Linear(hidden_size, num_classes)

    def forwar(self, x):
        # 设置初始隐藏状态和细胞状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(
            0), self.hidden_size).to(x.device)
        
        # 通过RNN层，得到输出和最后隐藏状态
        out, _ = self.run(x, (h0,c0))
        # 去最后一个时间步的输出，通过全连接层得到最终输出
        out = self.fc(out[:, -1, :])
        return out


# 实例化模型、损失函数和优化器
model = RNN(input_size, hidden_size, num_layers, num_classes)
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# Adam优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True)
test_dataset = datasets.MNIST(root='./data', train=False)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_sise, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_sise, shuffle=True)

# 训练模型
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        # 前向传播，得到预测输出
        outputs = model(inputs)
        # 计算损失值
        loss = criterion(outputs, labels)
        # 清空梯度缓存（因为PyTorch会累积梯度）
        optimizer.zero_grad()
        # 反向传播，计算梯度值
        loss.backward()
        # 根据梯度执行权重（执行优化步骤）
        optimizer.step()
        if (i + 1) % 100 == 0:
            # 每100个Batch打印一次损失值和准确率
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch +
                  1, num_epochs, i + 1, len(train_loader), loss.item()))

print('Training finshied.')

# 定义一个双向循环层，这里使用LSTM单元作为基础
class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.5):
        super(BiRNN, self).__init__()

        # 正向和反向的LSTM层
        self.rnn = nn.LSTM(input_size,  hidden_size,
                          num_layers, bidirectional=True, dropout=dropout)
        # 全连接层, 假设我们做分类任务，类别数量为output_size
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forwar(self, x, hidden=None):
        batch_sise = x.size(0)
        seq_length = x.size(1)
        total_hidden_size = 2 * self.rnn.hidden_size  # 双向所以是两个隐藏层大小

        # LSTM前向传播
        outputs, (hidden, cell) = self.run(x, hidden)
        # 合并正向、反向的隐藏状态，得到每个时间步的完整上下文表示
        outputs = outputs.contiguous().view(-1, total_hidden_size)
        # 通过全连接进行分类
        predictions = self.fc(outputs)
        # 将预测的数据恢复为原始的时间序列形状
        predictions = predictions.view(batch_sise, seq_length, -1)
        return predictions, hidden

# 模型实例化
model = BiRNN(input_size, hidden_size, output_size)

# 假设x是准备好的输入数据
inputs = torch.randn((batch_sise, sequence_length, input_size))
outputs, _ = model(inputs)
