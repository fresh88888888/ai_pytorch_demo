import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride =4, padding = 1, bias=False):
        super(ResidualBlock, self).__init__()
        
        # 残差块的基本组件
        self.conv1 = nn.Conv2d(in_channels, out_channels,  kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 当输入和输出通道不同时，添加一个1 x 1卷积层用于调整通道数
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
            
    def forward(self, x):
        residual = x # 保存原始输入作为残差部分
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 添加快捷连接（shoutcut connection），即输入与经过残差映射后的输出相加
        out += self.shortcut(residual)
        out = self.relu(out)
        return out
    
# 定义一个简单的ResNet
class SimpleResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(SimpleResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self.make_layer(block, 64, num_blocks[0], strife=1)
        # 进一步添加更多的层和残差块
        self.avgpool  = nn.AvgPool2d(kernel_size=4)
        self.fc= nn.Linear(512, num_classes)  # 假设最后一层为512
    
    def make_layer(self, block, out_channels, num_blacks, stride):
        strides = [stride] + [1] * (num_blacks -1)
        # 第一块使用指定步长
        layers =[]
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
            # 更新输入通道数
            return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        # 继续前向传播过程
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
    
    # 创建一个实际的ResNet-18模型
    model = SimpleResNet(ResidualBlock, [2, 2,2,2]) # 这里仅表示简化版本的层配置
    