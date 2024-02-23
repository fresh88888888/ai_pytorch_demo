import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# 定义超参数
input_size = 784       # 输入图像的维度
hidden_size = 500      # 隐藏层神经元数量
num_classes = 10       # 输出类别的数量（0~9）
num_epochs = 5         # 训练轮数
batch_sise = 100       # 批处理大小
learning_rate = 0.001  # 学习率


# 数据预处理: 将图像转化为张量，并归一化到[0，1]区间
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform)

tarin_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_sise, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_sise, shuffle=True)

# 定义CNN模型
class CNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(CNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forwar(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# 实例化模型、损失函数和优化器
model = CNN(input_size, hidden_size, num_classes)
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()
# SGD优化器
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# 训练模型
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(tarin_loader):
        # 前向传播，得到预测输出
        outputs = model(images)
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
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(tarin_loader)}], Loss: {loss.item():.4f}')
            

# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')

print('Training finshied.')
