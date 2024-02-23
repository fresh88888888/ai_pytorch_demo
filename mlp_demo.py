import torch
import torch.nn as nn
import torch.optim as optim
from torchtext import data, datasets


# 定义MLP模型
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forwar(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 数据预处理
TEXT = data.Field(tokenize='spacy', lower=True)
LABEL = data.LabelField(dtype= torch.float)

train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)

# 构建词汇表
MAX_VOCAB_SIZE = 64
TEXT.build_vocab(train_data, MAX_VOCAB_SIZE)
LABEL.build_vocab(train_data)

# 数据加载器
batch_size = 64
train_iterator, test_iterator = data.BucketIterator.splits(
    (train_data, test_data),
    batch_size = batch_size,
    device = device,
)

# 初始化模型、损失函数和优化器
model = MLP(input_dim=TEXT.vocab.vectors.shape[1], hidden_dim=512, output_dim=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters())

# 训练模型
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    for texts, labels in enumerate(train_iterator):
        # 清空梯度缓存（因为PyTorch会累积梯度）
        optimizer.zero_grad()
        outputs = model(texts)
        loss = criterion(outputs.squeeze(), labels.float())
        # 反向传播，计算梯度值
        loss.backward()
        # 根据梯度执行权重（执行优化步骤）
        optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# 测试模型
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for texts, labels in test_iterator:
        outputs = model(texts)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total}%')
