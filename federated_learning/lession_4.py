from logging import ERROR
from collections import OrderedDict
import logging
from logging import INFO
import warnings
from flwr.common import ndarrays_to_parameters, Context
from flwr.server import ServerAppComponents
from flwr.client import Client, ClientApp, NumPyClient
from flwr.common.logger import (ConsoleHandler, console_handler, FLOWER_LOGGER, LOG_COLORS,)
from logging import LogRecord
from flwr.server import ServerApp, ServerConfig
from flwr.simulation import run_simulation
from flwr_datasets import FederatedDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr.client.mod import adaptiveclipping_mod
from flwr.server.strategy import (DifferentialPrivacyClientSideAdaptiveClipping, FedAvg,)


# Customize logging for the course.
class InfoFilter(logging.Filter):
    def filter(self, record):
        return record.levelno == INFO


FLOWER_LOGGER.removeHandler(console_handler)
warnings.filterwarnings("ignore")

# To filter logging coming from the Simulation Engine
# so it's more readable in notebooks
backend_setup = {"init_args": {"logging_level": ERROR, "log_to_driver": True}}


class ConsoleHandlerV2(ConsoleHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def format(self, record: LogRecord) -> str:
        """Format function that adds colors to log level."""
        if self.json:
            log_fmt = "{lvl='%(levelname)s', time='%(asctime)s', msg='%(message)s'}"
        else:
            log_fmt = (
                f"{LOG_COLORS[record.levelname] if self.colored else ''}"
                f"%(levelname)s {'%(asctime)s' if self.timestamps else ''}"
                f"{LOG_COLORS['RESET'] if self.colored else ''}"
                f": %(message)s"
            )
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


console_handlerv2 = ConsoleHandlerV2(timestamps=False, json=False, colored=True,)
console_handlerv2.setLevel(INFO)
console_handlerv2.addFilter(InfoFilter())
FLOWER_LOGGER.addHandler(console_handlerv2)


DEVICE = torch.device("cpu")
transforms = Compose([ToTensor(), Normalize((0.5,), (0.5,))])


def normalize(batch):
    batch["image"] = [transforms(img) for img in batch["image"]]
    return batch


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


def train_model(net, trainloader, epochs: int = 1):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())
    net.train()

    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def evaluate_model(net, testloader):
    net.to(DEVICE)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += ((torch.max(outputs.data, 1)[1] == labels).sum().item())
    accuracy = correct / len(testloader.dataset)
    return float(loss), float(accuracy)


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)


def get_weights(net):
    ndarrays = [val.cpu().numpy() for _, val in net.state_dict().items()]
    return ndarrays


def load_data(partition_id):
    fds = FederatedDataset(dataset="mnist", partitioners={"train": 10})
    partition = fds.load_partition(partition_id)

    traintest = partition.train_test_split(test_size=0.2, seed=42)
    traintest = traintest.with_transform(normalize)
    trainset, testset = traintest["train"], traintest["test"]

    trainloader = DataLoader(trainset, batch_size=64, shuffle=True)
    testloader = DataLoader(testset, batch_size=64)
    return trainloader, testloader


class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, testloader):
        self.net = net
        self.trainloader = trainloader
        self.testloader = testloader

    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        train_model(self.net, self.trainloader)
        return get_weights(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = evaluate_model(self.net, self.testloader)
        return loss, len(self.testloader), {"accuracy": accuracy}


def client_fn(context: Context) -> Client:
    net = SimpleModel()
    partition_id = int(context.node_config["partition-id"])
    trainloader, testloader = load_data(partition_id=partition_id)
    return FlowerClient(net, trainloader, testloader).to_client()


net = SimpleModel()
params = ndarrays_to_parameters(get_weights(net))


def server_fn(context: Context):
    fedavg_without_dp = FedAvg(fraction_fit=0.6, fraction_evaluate=1.0, initial_parameters=params,)
    fedavg_with_dp = DifferentialPrivacyClientSideAdaptiveClipping(fedavg_without_dp, noise_multiplier=0.3, num_sampled_clients=6,)

    # Adjust to 50 rounds to ensure DP guarantees hold
    # with respect to the desired privacy budget
    config = ServerConfig(num_rounds=5)

    return ServerAppComponents(strategy=fedavg_with_dp, config=config,)


client = ClientApp(client_fn, mods=[adaptiveclipping_mod], )
server = ServerApp(server_fn=server_fn)

run_simulation(server, client, backend_config=backend_setup)
