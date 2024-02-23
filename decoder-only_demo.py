import torch
import torch.nn as nn

# 假设我们有一个词表，大小(vocab_size)
vocab_size = 10000

# 定义Decoder隐藏层维度
hidden_size = 512

# 定义自注意力机制所需要的参数
num_attention_heads = 8
attention_dropout = 0.1

#Decoder类定义
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        # 自注意力子层
        self.self_attention = nn.MultiheadAttention(hidden_size, num_attention_heads, dropout=attention_dropout)
        
        # 前馈神经网络子层
        self.feedforward_network = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        
        # 层归一化和残差连接
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        # Dropout用于减少过拟合
        self.dropout = nn.Dropout(0.1)
        
        def forward(self, input_tokens):
            # 自注意力步骤
            query, key, value = input_tokens, input_tokens, input_tokens
            attention_output, _= self.self_attention(query, key, value)
            attention_output, _ =self.dropout(attention_output)
            out1 = self.norm1(attention_output + input_tokens)       # 残差连接
            
            # 前馈神经网络处理
            ff_output = self.feedforward_network(out1)
            out2 = self.norm2(ff_output + out1)

            return out2

# 创建一个decoder实例
decoder = Decoder()

# 假设我们有随机初始化的输入
input_seq = torch.randn(32, 64, hidden_size)

# 运行Decoder
output_seq = decoder(input_seq)

            