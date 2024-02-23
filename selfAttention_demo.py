import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init(self, embed_size):
        super(SelfAttention, self).__init__()
        
        self.embed_size = embed_size
        self.liner = nn.Linear(embed_size, embed_size)
        self.gmma = nn.Parameter(torch.tensor(0.), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, inputs):
        # 计算query、key、value
        query = self.liner(inputs);
        key = inputs
        value = inputs
        
        # 计算query和Key之间的相关性得分
        scores = torch.bmm(query, key.transpose(1, 2))
        scores = self.gmma + scores    # 添加gmma用于缩放，使其落在合适的范围内
        
        # 对相关性得分应用softmax函数，得到Attention权重
        attention_weights = self.softmax(scores)
        
        # 使用attention权重和value计算输出
        output = torch.bmm(value, attention_weights.transpose(1, 2))
        
        return output  # 返回注意力机制的输出结果
    

        