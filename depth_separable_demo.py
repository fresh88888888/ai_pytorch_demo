import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义深度可分离卷积层
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channles, out_channles, kernel_size=3, stride=1, padding=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # 创建深度卷积层，每个卷积核与输入特征图的一个通道进行计算
        self.depthwise_conv = nn.Conv2d(in_channles, in_channles, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channles)
        
        # 创建逐点卷积层，对深度卷积的输出进行独立的缩放和偏移
        self.pointwise_conv = nn.Conv2d(in_channles, out_channles, kernel_size=1, stride=1, padding=0)
        
    def forward(self, x):
        # 首先进行蛇毒卷积
        x_depthwise = self.depthwise_conv(x)
        # 然后进行逐点卷积
        x_pointwise = self.pointwise_conv(x_depthwise)
        
        return x_pointwise
    
# 创建一个简单的深度可分离卷积网络
class SimpleSeparableNet(nn.Module):
    def __init__(self, input_channles=3, out_channles=64, kernel_size=3, stride=1, padding=1):
        super(SimpleSeparableNet, self).__init__()

        # 首先应用深度可分离卷积
        self.ds_conv = DepthwiseSeparableConv(input_channles, out_channles, kernel_size = kernel_size, stride=stride, padding=padding)
        
        # 然后接一个ReLU激活函数
        self.relu = nn.ReLU
        
        # 最后接一个池化层，这里用的是2x2的最大池化
        self.pool = nn.MaxPool2d(kernel_size= 2, stride=2)
        
    def forward(self, x):
        # 先通过深度可分离卷积
        x = self.ds_conv(x)
        # 然后通过ReLu激活函数
        x = self.relu(x)
        # 最后通过池化层
        x = self.pool(x)
        
        return x

#创建深度可分离卷积网络
model = SimpleSeparableNet()

# 创建一个随机张量作为输入
input_tensor = torch.randn(1,3,32,32)

# 通过网络前向传播输入张量
output_tensor = model(input_tensor)
