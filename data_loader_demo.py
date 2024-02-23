import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# 加载本地数据的方法
def load_data(data_path):
    return []

# 定义一个简单的线性模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()

        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)
    
# 定义一个自定义的数据集类（继承自 torch.utils.data.Dataset）
class CustomerDataset(Dataset):
    def __init__(self, data_path, transform = None):
        # 初始化函数，载入数据并设置与处理操作
        self.data = load_data(data_path)
        self.transform = transform
        
    def __len__(self):
        # 返回数据集中样本的数量
        return len(self.data)
    
    def __getitem__(self, index):
        # 根据索引获取单个样本及其对应标签
        sample, label = self.data[index]
        if self.transform:
            sample = self.transform(sample)  # 数据预处理
        return sample, label                 # 返回图像和对应的标签
    
# 实例化自定义数据集对象
data_transforms = transforms.Compose([
    transforms.Resize(32, 32),      # 对图像进行缩放
    transforms.ToTensor(),          # 将图像数据转化为张量
    transforms.Normalize(mean=[0.5], std=[0.5]),  # 对输入数据进行归一化
])

dataset = CustomerDataset('./data', transform=data_transforms)

# 设置DataLoader参数
batch_size = 64  # min-batch 的大小
shuffle = True   # 是否在每个epoch开始时对数据进行随机排序
num_works = 4    # 同时工作的数据加载器子进程数量
num_epochs = 5   # 训练轮数

# 创建DataLoader实例
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_works)

# 初始化模型，优化器以及损失函数
model = SimpleModel()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()
# 交叉熵损失函数
criterion = nn.CrossEntropyLoss()

# 现在可以使用DataLoader在训练循环中加载批量数据
for epoch in range(num_epochs):
    # 训练多个周期(epoch)
    for batch_idx, (inputs, labels) in enumerate(data_loader):
        # 模型前向传播，反向传播及优化步骤
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 打印当前批次的损失或其他信息
        print(f'Epoch: {epoch}, Batch Index: {batch_idx}, Loss: {loss.item()}')

