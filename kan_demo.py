from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from kan import KAN
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import torch
import numpy as np

dataset = {}
train_input, train_label = make_moons(n_samples=1000, shuffle=True, noise=0.1, random_state=None)
test_input, test_label = make_moons(n_samples=1000, shuffle=True, noise=0.1, random_state=None)

dataset['train_input'] = torch.from_numpy(train_input)
dataset['test_input'] = torch.from_numpy(test_input)
dataset['train_label'] = torch.from_numpy(train_label)
dataset['test_label'] = torch.from_numpy(test_label)

# X = dataset['train_input']
# y = dataset['train_label']
# plt.scatter(X[:, 0], X[:, 1], c=y[:])
# plt.show()


model = KAN(width=[2, 2], grid=3, k=3)


def train_acc():
    return torch.mean((torch.argmax(model(dataset['train_input']), dim=1) == dataset['train_label']).float())


def test_acc():
    return torch.mean((torch.argmax(model(dataset['test_input']), dim=1) == dataset['test_label']).float())


results = model.train(dataset, opt="LBFGS", steps=20, metrics=(
    train_acc, test_acc), loss_fn=torch.nn.CrossEntropyLoss())
lib = ['x', 'x^2', 'x^3', 'x^4', 'exp', 'log', 'sqrt', 'tanh', 'sin', 'abs']
model.auto_symbolic(lib=lib)

# 让我们通过查看训练和测试的准确性来检查该公式的准确性。
formula1, formula2 = model.symbolic_formula()[0]


def acc(formula1, formula2, X, y):
    batch = X.shape[0]
    correct = 0
    for i in range(batch):
        logit1 = np.array(formula1.subs('x_1', X[i, 0]).subs('x_2', X[i, 1])).astype(np.float64)
        logit2 = np.array(formula2.subs('x_1', X[i, 0]).subs('x_2', X[i, 1])).astype(np.float64)
        correct += (logit2 > logit1) == y[i]

    return correct/batch


# Print Accuracy
print('train acc of the formula:', acc(formula1, formula2, dataset['train_input'], dataset['train_label']))
print('test acc of the formula:', acc(formula1, formula2, dataset['test_input'], dataset['test_label']))

# 这样就完成了KAN模型。现在，让我们使用scikit-learn的 MLPClassifier模型为此数据集创建一个MLP分类器。首先，我们需要导入必要的库。
# 我们需要从之前创建的数据集中提取数据以适合该模型。我们使用.numpy()将张量转换为NumPy数组。
X_train = dataset['train_input'].numpy()
y_train = dataset['train_label'].numpy()
X_test = dataset['test_input'].numpy()
y_test = dataset['test_label'].numpy()

# 请记住将特征缩放作为我们增强模型性能过程的一部分。
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 现在，我们训练我们的神经网络。在这种情况下，alpha表示L2正则化项的强度，max_iter指定最大迭代次数。定义模型后，我们继续将其拟合到数据。
clf = MLPClassifier(alpha=1, max_iter=1000)
clf.fit(X_train, y_train)

# 现在所有任务都已完成，让我们回顾一下训练和测试的准确性，以评估MLP的性能。
train_accuracy = clf.score(X_train, y_train)
test_accuracy = clf.score(X_test, y_test)

# Print the accuracies
print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")
