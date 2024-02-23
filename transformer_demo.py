import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer

visual_input_size = 10
textual_input_size = 20
# 假设我们有两个模态，每个模态有各自的特征提取器提取出固定长度的向量表示
class ModelFeatureExtractor(nn.Module):
    def __init__(self, input_size, output_size):
        super(ModelFeatureExtractor, self).__init__()
        
        self.model_net = nn.Linear(input_size, output_size)
        
    def forward(self, model_input):
        return self.model_net(model_input)
    
# 定义多模态融合模块，这里使用了共享的Transformer编码器层
class MultiModelFusionTransformer(nn.Module):
    def __init__(self, model_size, fusion_dim, num_heads, num_layers, dropout):
        super(MultiModelFusionTransformer, self).__init__()
        
        # 分别为两个创建特征提取器
        self.visual_extractor = ModelFeatureExtractor(visual_input_size, fusion_dim)
        self.textual_extractor = ModelFeatureExtractor(textual_input_size, fusion_dim)

        # 创建一个共享的Transformer编码器核心
        fused_representation = self.trnsformer_encoder(combined_features)
        
        return fused_representation

if __name__ == '__main__':
    # 假设我们有一些预处理后的视觉和文本数据
    visual_data = torch.randn((batch_size, visual_seq_length, visual_feature_size))
    textual_data = torch.randn(
        (batch_size, textual_seq_length, textual_feature_size))

    model = MultiModelFusionTransformer(
        model_size=(visual_feature_size, textual_feature_size),
        fusion_dim = 256,    # 融合的维度大小
        num_heads = 8,       # 自注意力头数
        num_layers = 2,      # 编码器层数
        dropout=0.1          # 正则化dropout比例
    )
    
    # 计算融合后的特征
    fused_representations = model(visual_data, textual_data)
    
    
class TransformerModel(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, num_classes, nhead, nhid, nlayers, dropout):
        """
        初始化Transformer模型
        
        参数:
            vocabulary_size (int): 词汇表大小
            embedding_dim (int): 
            num_classes (int: 输出类别数
            nhead (int): 多头自注意力机制中的头数
            nhid (int): 隐藏层维度
            nlayers (int): Transformer编码器和解码器层的数量
            dropout (float): dropout 概率
        """
        super(TransformerModel, self).__init__()

        # 嵌入层，将词汇表中的词转换为固定维度的向量
        self.encoder = nn.Embedding(
            vocabulary_size, embedding_dim=embedding_dim)
        # Transformer结构， 包含自注意力机制和位置编码
        self.transformer = nn.Transformer(d_model=embedding_dim, nhead=nhead, num_encoder_layers=nlayers, num_encoder_layers=nlayers, dropout=dropout)
        # 输出层，将Transformer的输出转化为类别概率分布
        self.decoder = nn.Linear(embedding_dim, num_classes)

        # 初始化权重参数
        self.init_weights()
        
    def init_weights(self):
        '''
        初始化权重参数
        '''
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        '''
        前向传播过程
        '''
        # 将输入数据嵌入到固定维度的向量中
        embedded = self.encoder(src)

        # 通过Transformer结构进行自注意力和位置编码处理
        output = self.transformer(embedded)

        # 将Transformer的输出转化为类别概率分布
        output = self.decoder(output)

        return output   # [batch_size, num_classes]
    

# 定义Transformer模型
class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers, num_heads):
        super(Transformer, self).__init__()

        self.embedding = nn.Embedding(input_dim, hidden_dim)
        
        self.transformer_layers = nn.ModuleList([nn.TransformerEncoderLayer(hidden_dim, num_heads) for _ in range(num_layers)])
        
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.transformer_layers:
            x = layer(x)
            x = self.fc(x)
            
            return x
    

# 超参数设置
input_dim = 1000
output_dim = 1000
hidden_dim = 512
num_layers = 3
num_heads = 8

# 实例化参数
model = Transformer(input_dim, output_dim, hidden_dim, num_layers, num_heads)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 模拟输入数据
input_data = torch.randint(0,input_dim, (32, 128))  # 32个样本，每个样本128个特征
output_data = torch.randint(0, output_dim, (32, 128))  # 32个样本，每个样本长度为128

# 前向传播
outputs = model(input_data)

# 计算损失
loss = criterion(outputs.view(-1, output_dim), output_data.view(-1))

# 反向传播和优化
optimizer.zero_grad()
loss.backward()
optimizer.step()

print("Loss: ", loss.item())


# 假设我们有一个训练好的Transformer模型和对应的分词器
model = TransformerModel.from_pretrained('your_transformer_model')
tokenizer = TransformerTokenizer.from_pretrained('your_transformer_model')

# 假设我们有用户的历史行为序列数据
use_behavior = ['user read book1', 'user read book2', 'user read book3']

# 将行为序列转化为模型可以处理的输入格式
encoder_sequences = tokenizer([use_behavior], padding=True, truncation= True, return_tensors='pt')

#通过Transformer模型获取用户行为序列的上下文表示
contextual_embeddings = model(**encoder_sequences)[0]

# 现在我们可以用这个上下文表示来进行后续的推荐任务，例如：找最相关的书籍
# 这里仅做简化示意，实际应用中会更复杂，可能包括与商品库中的所有商品嵌入计算相似度等步骤
predicted_book_embedding = model.book_embedding('book4')

# 计算预测的商品与用户兴趣的匹配程度（这里仅做简单示例）
similar_score = torch.cosine_similarity(contextual_embeddings[:, -1], predicted_book_embedding)

# 根据相似度得分进行推荐
if similar_score > threshold:
    print(f'推荐书籍：book4， 与用户兴趣匹配为：{similar_score.item()}')
    
