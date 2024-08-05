from logging import ERROR, INFO
import torch
import torch.nn as nn
import torch.utils
from torch.utils.data import Subset, DataLoader, random_split
import torch.optim as optim
import torch.utils.data
from torchvision import transforms, datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import logging
from flwr.common.logger import console_handler, log
from flwr.common import Metrics, NDArrays, Scalar
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common import ndarrays_to_parameters, Context
from flwr.server import ServerApp, ServerConfig
from flwr.server import ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional

backend_setup = {"init_args": {"logging_level": ERROR, "log_to_driver": False}}
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, ), (0.5, ))])


def normalize(batch):
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch


class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == INFO


console_handler.setLevel(INFO)
console_handler.addFilter(InfoFilter())


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(784, 128)
        self.relu = nn.ReLU()
        self.out = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.relu(x)
        x = self.out(x)

        return x


def train_model(model, train_set):
    batch_size = 64
    num_epochs = 10
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.01
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {running_loss / len(train_loader)}")

    print("Training complete")


def evaluate_model(model, test_set):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    total_loss = 0

    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)
            total_loss += loss.item()

    accuracy = correct / total
    average_loss = total_loss / len(test_loader)

    return average_loss, accuracy


def include_digits(dataset, included_digits):
    including_indices = [idx for idx in range(len(dataset)) if dataset[idx][1] in included_digits]

    return torch.utils.data.Subset(dataset, including_indices)


def exclude_digits(dataset, excluded_digits):
    including_indices = [idx for idx in range(len(dataset)) if dataset[idx][1] not in excluded_digits]
    return torch.utils.data.Subset(dataset, including_indices)


def plot_distribution(dataset, title):
    labels = [data[1] for data in dataset]
    unique_labels, label_counts = torch.unique(torch.tensor(labels), return_counts=True)
    plt.figure(figsize=(4, 2))

    counts_dict = {label.item(): count.item() for label, count in zip(unique_labels, label_counts)}

    all_labels = np.arange(10)
    all_label_counts = [counts_dict.get(label, 0) for label in all_labels]

    plt.bar(all_labels, all_label_counts)
    plt.title(title)
    plt.xlabel("Digit")
    plt.ylabel("Count")
    plt.xticks(all_labels)
    plt.show()


def compute_confusion_matrix(model, test_set):
    # Initialize lists to store true labels and predicted labels
    true_labels = []
    predicted_labels = []

    # Iterate over the test set to get predictions
    for Image, label in test_set:
        # Forward pass through the model to get predictions
        output = model(Image.unsqueeze(0))
        _, predicted = torch.max(output, 1)

        # Append true and predicted labels to lists
        true_labels.append(label)
        predicted_labels.append(predicted.item())

    # Convert lists to numpy arrays
    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    return cm


def plot_confusion_matrix(cm, title):
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, cmap="Blues", fmt='d', linewidths=.5)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


train_set = datasets.MNIST("./MNIST_data/", download=True, train=True, transform=transform)
total_length = len(train_set)
split_size = total_length // 3
torch.manual_seed(42)
part_1, part_2, part_3 = random_split(train_set, [split_size] * 3)

part_1 = exclude_digits(part_1, excluded_digits=[1, 3, 7])
part_2 = exclude_digits(part_2, excluded_digits=[2, 4, 6])
part_3 = exclude_digits(part_3, excluded_digits=[4, 6, 9])

# plot_distribution(part_1, 'Part 1')
# plot_distribution(part_2, 'Part 2')
# plot_distribution(part_3, 'Part 3')
train_set = [part_1, part_2, part_3]

# model1 = SimpleModel()
# train_model(model1, part_1)

# model2 = SimpleModel()
# train_model(model2, part_2)

# model3 = SimpleModel()
# train_model(model3, part_3)

testset = datasets.MNIST(
    "./MNIST_data/", download=True, train=False, transform=transform
)

testset_137 = include_digits(testset, included_digits=[1, 3, 7])
testset_246 = include_digits(testset, included_digits=[2, 4, 6])
testset_469 = include_digits(testset, included_digits=[4, 6, 9])

# _, accuracy1 = evaluate_model(model1, testset)
# _, accuracy1_on_137 = evaluate_model(model1, testset_137)
# print(f"Model 1-> Test Accuracy on all digits: {accuracy1:.4f}, "f"Test Accuracy on [1,3,7]: {accuracy1_on_137:.4f}")

# _, accuracy2 = evaluate_model(model2, testset)
# _, accuracy2_on_246 = evaluate_model(model2, testset_246)
# print(f"Model 2-> Test Accuracy on all digits: {accuracy2:.4f}, "f"Test Accuracy on [2,4,6]: {accuracy2_on_246:.4f}")

# _, accuracy3 = evaluate_model(model3, testset)
# _, accuracy3_on_469 = evaluate_model(model3, testset_469)
# print(f"Model 3-> Test Accuracy on all digits: {accuracy3:.4f}, "f"Test Accuracy on [4,6,9]: {accuracy3_on_469:.4f}")

# confusion_matrix_model1_all = compute_confusion_matrix(model1, testset)
# confusion_matrix_model2_all = compute_confusion_matrix(model2, testset)
# confusion_matrix_model3_all = compute_confusion_matrix(model3, testset)

# plot_confusion_matrix(confusion_matrix_model1_all, "model 1")
# plot_confusion_matrix(confusion_matrix_model2_all, "model 2")
# plot_confusion_matrix(confusion_matrix_model3_all, "model 3")

# Define training and testing in the pipeline


# Sets the parameters of the model
def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


# Retrieves the parameters from the model
def get_weights(net):
    ndarrays = [val.cpu().numpy() for _, val in net.state_dict().items()]

    return ndarrays


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader):
        super().__init__()
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    # Train the model
    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        epochs = config["local_epochs"]
        log(INFO, f"client train for {epochs} epochs.")
        train_model(self.net, self.trainloader, epochs)
        return get_weights(self.net), len(self.trainset), {}

    # Test the model
    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]):
        set_weights(self.net, parameters)
        loss, accuracy = evaluate_model(self.net, self.testset)
        return loss, len(self.testset), {"accuracy": accuracy}


def client(context: Context) -> Client:
    net = SimpleModel()
    partition_id = int(context.node_config['partition-id'])
    client_train = train_set[int(partition_id)]
    client_test = testset

    return FlowerClient(net=net, trainset=client_train, testset=client_test).to_client()


# Create an instance of the ClientApp.
client = ClientApp(client_fn=client)


def evaluate(server_round, parameters, config):
    net = SimpleModel()
    set_weights(net, parameters)

    _, accuracy = evaluate_model(net, testset)
    _, accuracy_137 = evaluate_model(net, testset_137)
    _, accuracy_246 = evaluate_model(net, testset_246)
    _, accuracy_469 = evaluate_model(net, testset_469)

    log(INFO, "test accuracy on all digitss: %.4f", accuracy)
    log(INFO, "test accuracy on [1,3,7]: %.4f", accuracy_137)
    log(INFO, "test accuracy on [2,4,6]: %.4f", accuracy_246)
    log(INFO, "test accuracy on [4,6,9] %.4f", accuracy_469)

    if server_round == 3:
        cm = compute_confusion_matrix(net, testset)
        plot_confusion_matrix(cm, "Final Global Model")


def load_data(partition_id):
    fds = FederatedDataset(dataset="mnist", partitioners={"train": 5})
    partition = fds.load_partition(partition_id)

    traintest = partition.train_test_split(test_size=0.2, seed=42)
    traintest = traintest.with_transform(normalize)
    trainset, testset = traintest['train'], traintest['test']

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(testset, batch_size=64)

    return train_loader, test_loader


def fit_config(server_round: int):
    config_dict = {"local_epochs": 2 if server_round < 3 else 5}

    return config_dict


# 策略：联邦平均
net = SimpleModel()
params = ndarrays_to_parameters(get_weights(net))


def server(context: Context):
    strategy = FedAvg(min_fit_clients=5, fraction_evaluate=0.0, initial_parameters=params, on_fit_config_fn=fit_config)
    config = ServerConfig(num_rounds=3)

    return ServerAppComponents(strategy=strategy, config=config)


# 创建SreverApp的实例
server = ServerApp(server_fn=server)

# 开始训练
run_simulation(server_app=server, client_app=client, num_supernodesq=3, backend_config=backend_setup)
