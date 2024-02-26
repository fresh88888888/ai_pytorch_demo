import torch
import torch.nn.functional as F

# 假设我们有两个向量a和b
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# 计算两个向量的余弦相似度
consine_similarity = F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0), dim=1)

print(f'The cosine similarity between a and b is: {consine_similarity.item()}')

# 假设我们有一个向量集合，我们想找到与向量a最相似的向量
vector_collection = torch.tensor([[7.0, 8.0, 9.0], [1.5, 2.5, 3.5], [0.5, 1.5, 2.5]])

# 计算向量a与向量集合中每个向量的余璇相似度
similarities = F.cosine_similarity(a.unsqueeze(0), vector_collection, dim=1)

# 找到最相似的向量
most_similar_vector_idex = torch.argmax(similarities).item()
most_similar_vector = vector_collection[most_similar_vector_idex]

print(f'The most similar vector to a is: {most_similar_vector}')
