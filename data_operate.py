import torch as torch

x = torch.arange(12)
print(x)
print(x.shape)
print(x.numel())
print(x.reshape(3, 4))
print(x.reshape(-1, 4))
print(torch.zeros((2, 3, 4)))
print(torch.ones((2, 3, 4)))
print(torch.randn(3, 4))
print(torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]]))

x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
print(f"x+y={x +y}\nx-y={x - y}\nx*y={x*y}\nx/y={x/y}\nx^y={x**y}")
print(torch.exp(x))
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0))
print(torch.cat((X, Y), dim=1))
print(X == Y)
print(X.sum())

a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(f"{a}\n{b}")
print(a + b)

print(f"X[-1]:{X[-1]}\nX[1:3]:{X[1:3]}")
X[1, 2] = 9
print(X)
X[0:2, :] = 12
print(X)

before = id(Y)
Y = Y + X
print(id(Y) == before)

Z = torch.zeros_like(Y)
print('id(Z):', id(Z))
Z[:] = X + Y
print('id(Z):', id(Z))

before = id(X)
X += Y
print(id(X) == before)

A = X.numpy()
B = torch.tensor(A)
print(f"{type(A)}\n{type(B)}")
a = torch.tensor([3.5])
print(f'a:{a}, a.item:{a.item()}, int(a):{int(a)}, float(a):{float(a)}')
print(dir(torch.distributions))
