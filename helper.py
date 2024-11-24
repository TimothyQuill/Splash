"""
This file abstracts away all the processes of training
the model, allowing train.py to isolate the key steps.
"""

import os
import json
import torch

from torch.utils.data import DataLoader, TensorDataset
from hyperparameters import *


def create_data_loader(x, y):
    """
    Creates a DataLoader for batching and shuffling the input data.

    Args:
        x (torch.Tensor): Input tensor of shape (num_samples, ...).
        y (torch.Tensor): Target tensor of shape (num_samples, ...).

    Returns:
        DataLoader: DataLoader object for the dataset with specified batch size.
    """
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def create_dir():
    """
    Creates the directory specified in SAV_DIR if it doesn't exist.

    Args:
        None

    Returns:
        None
    """
    os.makedirs(SAV_DIR, exist_ok=True)


def minmax_scaling(tensor, minimum_value=1, maximum_value=1_000):
    """
    Scales the values of a tensor to the range [0, 1].

    Args:
        tensor (torch.Tensor): Input tensor to be scaled.
        minimum_value (float): Minimum expected value (default: 1).
        maximum_value (float): Maximum expected value (default: 1,000).

    Returns:
        torch.Tensor: Tensor scaled to the range [0, 1].
    """
    if minimum_value == maximum_value:
        raise ValueError("All values in the tensor are identical. Scaling is undefined.")
    return (tensor - minimum_value) / (maximum_value - minimum_value)


def evaluate_model(model, test_loader, criterion):
    """
    Evaluates the model on the test dataset.

    Args:
        model (torch.nn.Module): Trained PyTorch model.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (torch.nn.Module): Loss function.

    Returns:
        float: Average loss on the test dataset.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in test_loader:
            src = src.view(-1, GIVEN_SEQ, N_CHANNELS)
            tgt = tgt.view(-1, TARGET_SEQ, N_CHANNELS)
            outputs = model(src, tgt)
            loss = criterion(outputs, tgt)
            total_loss += loss.item()
    return total_loss / len(test_loader)


def floats_to_ints(x):
    """
        Converts a nested structure of floats into integers.

        Args:
            x (list of lists of lists): A 3D list structure containing floats.

        Returns:
            list of lists of lists: A 3D list structure with all floats converted to integers.
        """
    x = x.tolist()
    return [[[int(num) for num in seq] for seq in ch] for ch in x]


def load_model(model):
    """
    Loads model weights from the directory specified in LOAD_DIR.

    Args:
        model (torch.nn.Module): PyTorch model instance.

    Returns:
        torch.nn.Module: Model with loaded weights.
    """
    model.load_state_dict(torch.load(LOAD_DIR))
    print(f"Model loaded from {LOAD_DIR}")
    return model


def normalise(x):
    """
    Normalizes a tensor to zero mean and unit variance along specified axes.

    Args:
        x (torch.Tensor): Input tensor of shape (...).

    Returns:
        torch.Tensor: Normalized tensor with zero mean and unit variance.
    """
    return (x - x.mean(axis=(0, 1, 3), keepdims=True)) / x.std(axis=(0, 1, 3), keepdims=True)


def reverse_minmax_scaling(scaled_tensor, minimum_value=1, maximum_value=1_000):
    """
    Reverses Min-Max scaling, restoring the tensor to its original range.

    Args:
        scaled_tensor (torch.Tensor): Tensor scaled to the range [0, 1].
        minimum_value (float): Min value of the tensor before scaling.
        maximum_value (float): Max value of the tensor before scaling.

    Returns:
        torch.Tensor: Tensor restored to its original range.
    """
    return scaled_tensor * (maximum_value - minimum_value) + minimum_value


def save_model(model, epoch):
    """
    Saves the model weights to a file named by the epoch.

    Args:
        model (torch.nn.Module): PyTorch model instance.
        epoch (int): Current epoch number.

    Returns:
        None
    """
    model_save_path = os.path.join(SAV_DIR, f"model_epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


def tensor_to_json(x):
    """
    Converts a PyTorch tensor to a JSON string.

    This function converts the input tensor to a Python list, which is then
    serialised into a JSON-formatted string.

    Args:
        x (torch.Tensor): The PyTorch tensor to be converted.

    Returns:
        str: A JSON-formatted string representing the tensor data.
    """
    return json.dumps(x)


def train_model(model, train_loader, optimizer, criterion):
    """
    Trains the model for one epoch using the training dataset.

    Args:
        model (torch.nn.Module): PyTorch model instance.
        train_loader (DataLoader): DataLoader for the training dataset.
        optimizer (torch.optim.Optimizer): Optimizer for model parameters.
        criterion (torch.nn.Module): Loss function.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    for src, tgt in train_loader:
        src = src.view(-1, GIVEN_SEQ, N_CHANNELS)
        tgt = tgt.view(-1, TARGET_SEQ, N_CHANNELS)

        optimizer.zero_grad()
        output = model(src, tgt)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def x_y_split(x):
    """
    Splits the input tensor into features (X) and targets (Y).

    Args:
        x (torch.Tensor): Input tensor of shape (..., GIVEN_SEQ + TARGET_SEQ).

    Returns:
        tuple: (X, Y), where:
            X (torch.Tensor): Features of shape (..., GIVEN_SEQ).
            Y (torch.Tensor): Targets of shape (..., TARGET_SEQ).
    """
    X = x[..., :GIVEN_SEQ]
    Y = x[..., GIVEN_SEQ:]
    return X, Y
