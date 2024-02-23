# Import the top-level package for core functionality
import torch

# Import neural network functionality
from torch import nn

# Import functional programing tool
import torch.nn.functional as F

# Import optimization functionality
import torch.optim as optim

# Import dataset functions
from torch.utils.data import TensorDataset, DataLoader

# Import evaluation metrics
import torchmetrics

from sklearn.datasets import load_wine
import pandas as pd

# ----------------------#

# Create tensor from list with tensor
tnsr = torch.tensor([1, 3, 6, 10])

# Get data type of tensor elements with .dtype
print(tnsr.dtype)  # Return torch.int64

# Get dimensions of tensor with .Size()
print(tnsr.shape)  # Returns torch.Size([4])

# Get memory location of tensor with .device
print(tnsr.device)  # Return the CPU or GPU

# Create a tensor of zeros with zeros()
tnsr_zrs = torch.zeros(2, 3)
print(tnsr_zrs)

# Create a random tensor with rand()
tnsr_rndm = torch.rand(size=(3, 4))  # Tensor has 3 rows, 4 columns
print(tnsr_rndm)

# Datasets and DataLoaders

# wine_data = load_wine()
# # Convert data to pandas dataframe
# wine_df = pd.DataFrame(wine_data.data, columns=wine_data.feature_names)


# def get_device() -> torch.device:
#     return torch.device('cpu')


# def df_to_tensor(df: pd.DataFrame) -> torch:
#     return torch.from_numpy(df.values).to(get_device())


# # Create a dataset from a pandas DataFrame with TensorDataset()
# dataset = df_to_tensor(wine_df)
# print(dataset)
# # Load the data in batches with DataLoader()
# dataloader = DataLoader(dataset=dataset, batch_size=10, shuffle=True)

# # Preprocessing

# # One-hot encode categorical variables with one_hot()
# # Returns tensor of 0s and 1s
# one_tensor = F.one_hot(torch.tensor([1, 2, 3]), num_classes=4)
# print(one_tensor)

# # Sequential Model Architecture

# # Create a linner layer with m inputs, n outputs with Linner()
# lnr = nn.Linear(2, 4)

# # Get weight of layer with .weight()
# print(lnr.weight)

# # Get bias of layer with .bias()
# print(lnr.bias)

# # Create a sigmoid activation layer for binary classification with Sigmoid()
# nn.Sigmoid()

# # Create a softmax activation layer for multi-class classification with Softmx()
# nn.Softmax(dim=-1)

# # Create a rectified linner unit activation layer to avoid saturation with ReLU()
# nn.ReLU()

# # Create a leakly rectified linner unit activation layer to avoid saturation with LeakReLU()
# nn.LeakyReLU(negative_slope=0.05)

# # Create a dropout layout to regularize and prevent overfitting with Dropout()
# nn.Dropout(p=0.5)

# # Create a squential model from layers
# model = nn.Sequential(nn.Linear(one_tensor, 2), nn.Linear(
#     2, 3), nn.Linear(3, 4), nn.Softmax(dim=1))


# # Fitting a model and calculating loss

# # Fit a model to input data with model where model is a variable created by, e.g., Sequential()
# predictions = model(input_data).double()

# # Get target values
# actual = torch.tensor(target_values).double()  # Returns tensor(x)

# # Calculate the mean-squared error loss for regression with MSELoss()
# mse_loss = nn.MSELoss()(predictions, actual)  # Returns tensor(x)

# # Calculate the L1 loss error loss for regression with SmoothL1Loss()
# l1_loss = nn.SmoothL1Loss()(predictions, actual)  # Returns tensor(x)

# # Calculate binary cross-entropy loss for binary classification with BCELoss()
# bce_loss = nn.BCELoss()(predictions, actual)  # Returns tensor(x)

# # Calculate cross-entropy loss for multi-class classification with CrossEntropyLoss()
# ce_loss = nn.CrossEntropyLoss()(predictions, actual)  # Returns tensor(x)

# # Calculate the gradients via backprogagation with .backward()
# loss.backward()

# # Working with Optimizers

# # Create a stochastic gradient descent optimizer with SGD(), setting learning rate and momentum
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.95)

# # Update neuron parameters with .step()
# optimizer.step()

# # The Training Loop

# # Set model to training mode
# model.train()
# # Set a loss criterion and an optimizer
# loss_criterion = nn.MSELoss()
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.95)
# # Loop over chunks of data in the training set
# for data in dataloader:
#     # Set the gradients to zero with .zero_grad()
#     optimizer.zero_grad()
#     # Get features and targets for current chunk of data
#     features, targets = data
#     # Run a "forward pass" to fit the model to the data
#     predictions = model(data)
#     # Calculate loss
#     loss = loss_criterion(predictions, targets)
#     # Calculate gradidents using backprogagation
#     loss.backward()
#     # Update the model parameters
#     optimizer.step()


# # Set model to evaluation mode
# model.eval()
# # Create accuracy metric with Accuracy()
# metric = torchmetrics.Accuracy(task='multiclass', num_classes=3)
# # Loop of chunks of data in the validation set
# for i, data in enumerate(dataloader, 0):
#     # Get features and targets for current chunk of data
#     features, targets = data
#     # Run a "forward pass" to fit the model to the data
#     predictions = model(data)
#     # Calculate accuracy over the batch
#     accuracy = metric(output, predictions.argmax(dim=-1))

# # Calculate accuracy over all the validation data
# accuracy = metric.compute()
# print(f'Accuracy on all data: {accuracy}')
# # Reset the metric for the next dataset(training or validation)
# metric.reset()

# # Transfer Learning and Fine-Tuning

# # Save a layer of a model to a file with save()
# torch.save(layer, 'layer.pth')

# # Load a layer of a model from a file with load()
# new_layer = torch.load('layer.pth')

# # Freeze the weight for layer 0 with .requires_grad
# for name, param in model.named_parameters():
#     if name == '0.weight':
#         param.requires_grad = False
