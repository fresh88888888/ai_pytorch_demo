import torch
import torch.nn as nn


# 定义一个自注意力模块
class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8):
        super(SelfAttention, self).__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # 对输入进行线性变化已创建查询键和值向量
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.ke_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        
        # 将嵌入维度分成多个头的大小
        self.num_heads = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, 'embed_dim must be divisible by num_heads'
        
        #初始化 归一化系数
        self.scaling = self.head_dim ** -0.5
    
    def forward(self, x):
        query = self.query_proj(x)
        key = self.ke_proj(x)
        value = self.value_proj(x)
        
        # 进行矩阵乘法和分隔
        query = query.reshape(-1, x.shape[1], self.num_heads, self.head_dim).transpose(1, 2)
        key = key.reshape(-1, x.shape[1], self.num_heads,
                              self.head_dim).transpose(1, 2)
        value = value.reshape(-1, x.shape[1], self.num_heads,
                              self.head_dim).transpose(1, 2)
        
        # 计算注意力权重
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / self.scaling
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)
        
        # 使用注意力权重加权求和得到上下文向量
        context = torch.matmul(attention_weights, value).transpose(1, 2).reshape(-1, x.shape[1], self.embed_dim)
        
        return context
    
# 定义一个简单的Transformer编解码器
class SimpleTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SimpleTransformerEncoderLayer, self).__init__()
        
        self.self_attntion = SelfAttention(embed_dim, num_heads)
        
        #在自注意力后还有一个前馈神经网络（FFN）
        self.feed_forward_network = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        
        #注意力后的残差层和归一化
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, inputs):
        #自注意力部分
        attn_output = self.self_attntion(inputs)
        out1 = self.norm1(inputs + attn_output)
        
        # 前馈神经网络部分
        ffn_output = self.feed_forward_network(out1)
        out2 = self.norm2(out1 + ffn_output)
        
        return out2

# 创建一个实例并应用到一些随机数据上
embed_dim = 512
num_heads = 8
model = SimpleTransformerEncoderLayer(embed_dim, num_heads)
# 假设我们有10个样本，每个样本包含20个特征的512维嵌入向量
inputs = torch.randn(10, 20, embed_dim)
output = model(inputs)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout = 0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attn= MultiHeadAttention(d_model, n_heads)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self_fc = nn.Linear(d_model, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self_dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None):
        # 输入src是经过位置编码后的词嵌入序列
        attn_output = self.self_attntion(src, src, src, attn_mask=atten_mask)
        attn_output = self.norm1(src + attn_output)
        out1 = self.norm1(src + attn_output)   # 残差连接归一化
        
        fc_out = self.fc(out1)
        fc_out = self.dropout2(fc_out)
        out2 = self.norm2(out1 + fc_out)       # 再次残差连接归一化
        
        return out2


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dropout = 0.1):
        super(TransformerDecoderLayer, self).__init__()

        self.self_attn= MultiHeadAttention(d_model, n_heads)
        self.enc_dec_attn = MultiHeadAttention(d_model, n_heads)
        # 注意力后的残差层和归一化
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self_fc = nn.Linear(d_model, d_model)

    def forward(self, tgt, memory, tgt_mask=None):
        # tgt 是待解码的目标序列词嵌入memory是编码器的输出，包含源语言的上下文信息
        self_attn_output, _ = self.self_attntion(tgt, tgt, tgt, attn_mask=tgt_mask)
        self_attn_output = self.dropout1(self_attn_output)
        out1 = self.norm1(tgt + self_attn_output)   # 目标自注意力部分的残差连接和归一化

        enc_dec_attn_output, _ = self.enc_dec_attn(out1, memory, attn_mask = memory_mask)
        enc_dec_attn_output = self.dropout2(enc_dec_attn_output)
        out2 = self.norm2(out1 + enc_dec_attn_output)  # 编码器-解码器注意力部分的残差连接和归一化
        
        fc_out = self.fc(out2)
        fc_out = self.dropout3(fc_out)
        out3 = self.norm3(out2 + fc_out)  # 全连接层后的残差连接和归一化

        return out3
