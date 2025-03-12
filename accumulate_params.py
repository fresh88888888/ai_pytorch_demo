import torch
import numpy as np


# Data format:
# [x1, x2] --> [y]
# [NUM_ROOMS, NUM_BATHS] --> [PRICE]
training_data = [
    [torch.tensor([6, 2], dtype=torch.float), torch.tensor([15], dtype=torch.float)],
    [torch.tensor([5, 2], dtype=torch.float), torch.tensor([12], dtype=torch.float)],
    [torch.tensor([5, 1], dtype=torch.float), torch.tensor([10], dtype=torch.float)],
    [torch.tensor([3, 1], dtype=torch.float), torch.tensor([7], dtype=torch.float)],
    [torch.tensor([2, 1], dtype=torch.float), torch.tensor([4.5], dtype=torch.float)],
    [torch.tensor([2, 0], dtype=torch.float), torch.tensor([4], dtype=torch.float)],
    [torch.tensor([1, 0], dtype=torch.float), torch.tensor([2], dtype=torch.float)],
]


# Define the model parameters
class ModelParameters:

    def __init__(self):
        self.w1 = torch.tensor(0.773, dtype=torch.float, requires_grad=True)
        self.w2 = torch.tensor(0.321, dtype=torch.float, requires_grad=True)
        self.b = torch.tensor(0.067, dtype=torch.float, requires_grad=True)


# We will use two training loops: the first one without gradient accumulation, and the second one with gradient accumulation.
params_no_accumulate = ModelParameters()
params_accumulate = ModelParameters()


def train_no_accumulate(params: ModelParameters, num_epochs: int = 10, learning_rate: float = 1e-3):
    """_summary_

    Args:
        params (ModelParameters): _description_
        num_epochs (int, optional): _description_. Defaults to 10.
        learning_rate (float, optional): _description_. Defaults to 1e-3.
    """
    print(f'Initial parameters: w1: {params.w1.item():.3f}, w2: {params.w2.item():.3f}, b: {params.b.item():.3f}')
    for epoch in range(1, num_epochs+1):
        for (x1, x2), y_target in training_data:
            # Calculate the output of the model
            z1 = x1 * params.w1
            z1.retain_grad()
            z2 = x2 * params.w2
            z2.retain_grad()
            y_pred = z1 + z2 + params.b
            y_pred.retain_grad()
            loss = (y_pred - y_target) ** 2

            # Calculate the gradients of the loss w.r.t. the parameters
            loss.backward()

            # Update the parameters (at each iteration)
            with torch.no_grad():
                # Equivalent to calling optimizer.step()
                params.w1 -= learning_rate * params.w1.grad
                params.w2 -= learning_rate * params.w2.grad
                params.b -= learning_rate * params.b.grad

                # Reset the gradients to zero
                # Equivalent to calling optimizer.zero_grad()
                params.w1.grad.zero_()
                params.w2.grad.zero_()
                params.b.grad.zero_()
        print(f"Epoch {epoch:>3} - Loss: {np.round(loss.item(),4):>10}")
    print(f'Final parameters: w1: {params.w1.item():.3f}, w2: {params.w2.item():.3f}, b: {params.b.item():.3f}')


# train_no_accumulate(params_no_accumulate)
def train_accumulate(params: ModelParameters, num_epochs: int = 10, learning_rate: float = 1e-3, batch_size: int = 2):
    print(f'Initial parameters: w1: {params.w1.item():.3f}, w2: {params.w2.item():.3f}, b: {params.b.item():.3f}')
    for epoch in range(1, num_epochs+1):
        for index, ((x1, x2), y_target) in enumerate(training_data):
            # Calculate the output of the model
            z1 = x1 * params.w1
            z1.retain_grad()
            z2 = x2 * params.w2
            z2.retain_grad()
            y_pred = z1 + z2 + params.b
            y_pred.retain_grad()
            loss = (y_pred - y_target) ** 2

            # We can also divide the loss by the batch size (equivalent to using nn.MSE loss with the paraemter reduction='mean')
            # If we don't divide by the batch size, then it is equivalent to using nn.MSE loss with the parameter reduction='sum'

            # Calculate the gradients of the loss w.r.t. the parameters
            # If we didn't call zero_() on the gradients on the previous iteration, then the gradients will accumulate (add up) over each iteration
            loss.backward()

            # Everytime we reach the batch size or the end of the dataset, update the parameters
            if (index + 1) % batch_size == 0 or index == len(training_data) - 1:
                with torch.no_grad():
                    # Equivalent to calling optimizer.step()
                    params.w1 -= learning_rate * params.w1.grad
                    params.w2 -= learning_rate * params.w2.grad
                    params.b -= learning_rate * params.b.grad

                    # Reset the gradients to zero
                    # Equivalent to calling optimizer.zero_grad()
                    params.w1.grad.zero_()
                    params.w2.grad.zero_()
                    params.b.grad.zero_()

        print(f"Epoch {epoch:>3} - Loss: {np.round(loss.item(),4):>10}")
    print(f'Final parameters: w1: {params.w1.item():.3f}, w2: {params.w2.item():.3f}, b: {params.b.item():.3f}')


train_accumulate(params_accumulate)
