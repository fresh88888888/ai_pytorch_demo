import torch

# 创建一个张量，并设置requires_grad True表示关心它的梯度
x = torch.tensor(2.0, requires_grad=True)
print(f'原始张量x: {x}')

# 动态计算过程开始，首先进行一个乘法运算
y = x * 2
print(f'张量y是x的2倍: y = {y}')

# 接着进行另一个乘法运算
z = y * 3
print(f'张量z是y的3倍：z = {z}')

# 此时，尽管我们没有明确定义一个固定的计算图，但PyTorch内部已经记录了x到z的完整计算路径
# 计算损失（这里假设z是我们要优化的目标函数的一部分）
loss = z.sum()

# 反向传播以计算梯度
loss.backward()     # 这会自动沿着动态生成的计算图反向传播求梯度

# 输出x的梯度，它使整个链式法则计算的结果
print(f'张量x的梯度：x.grad = {x.grad}')

# x.grad应该是6， 因为dz/dx = dz/dy * dy/dx = 3 * 2

