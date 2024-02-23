import torch
import torch.nn as nn

# 定义基础残差块
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None) :
        super(BasicBlock, self).__init__()
        
        # 第一个卷积层即归一化层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)
        
        # 第二个卷积层及归一化层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 采样操作
        self.downsample = downsample
        # 存储当前块的步长信息
        self.stride = stride
        
    def forward(self, x):
        # 记录原始输入以用于残差计算
        residual = x
        
        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 1000):
        super(ResNet, self).__init__()
        
        # 初始化通道数
        self.in_channels  = 64
        
        # 首先定义网络的第一层：7 x 7卷积、BN、ReLU和最大池化层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # 创建各个阶段的残差模块
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        # 平均池化层，将特征图转换为单个值
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

