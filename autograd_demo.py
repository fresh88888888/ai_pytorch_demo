import torch

# 创建一个需要计算梯度的Tensor
x = torch.ones(2,2,requires_grad=True)
print('x: ', x)

# 对Tensor进行操作
y = x + 2
z = y * y * 3
out = z.mean()

print('Operations on x: y = x + 2, z = y * y * 3, out = z.mean()')
print('y: ', y)
print('z: ', z)
print('out (loss): ', out)

# 计算梯度， 由于out是标量， 所以不需要指定grad_variables
out.backward()

# 查看x的梯度，即out关于x的梯度
print("Gradient of out with respect to x: ")
print(x.grad)

# 使用梯度更新参数（实际中会用优化器如SGD更新）
learning_rate = 0.1
x.data -= learning_rate * x.grad

# 在更新参数后， 需要清零梯度，因为下一轮迭代的梯度是基于当前参数的新梯度
x.grad.zero_()
