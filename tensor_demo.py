import torch

# 定义一个一维张量（向量）
tensor1 = torch.tensor([1,2,3,4])

# 定义一个二维张量（矩阵）
tensor2 = torch.tensor([[1,2],[3,4],[5,6]])

# 定义一个三维张量
tensor3 = torch.tensor([[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]])

# 查看张量的形状
print(tensor1.shape)
print(tensor2.shape)
print(tensor3.shape)

