"""
This file abstracts away all the processes of training
the model, allowing train.py to isolate the key steps.
"""

import os
import torch

from torch.utils.data import DataLoader, TensorDataset
from hyperparameters import *


def create_data_loader(x, y):
    """
    Plot training and testing loss curves and save the figure.

    Args:
        x (torch.Tensor): Data of shape (num_batches, batch_size,
            channels, given_sequence_len)
            train (list or array-like): Training loss values.
            test (list or array-like): Testing loss values.
            save_path (str): Path to save the plot image. Defaults to 'loss.jpg'.
    Returns:
        None
    """
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def create_dir():
    os.makedirs(SAV_DIR, exist_ok=True)


def evaluate_model(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, tgt in test_loader:
            src = src.view(-1, GIVEN_SEQ, N_CHANNELS)  # Flatten batch and batch size
            tgt = tgt.view(-1, TARGET_SEQ, N_CHANNELS)
            outputs = model(src, tgt)
            loss = criterion(outputs, tgt)
            total_loss += loss.item()
    return total_loss / len(test_loader)


def load_model(model):
    model.load_state_dict(torch.load(LOAD_DIR))
    model.eval()
    print(f"Model loaded from {LOAD_DIR}")
    return model


def normalise(x):
    return (x - x.mean(axis=(0, 1, 3), keepdims=True)) / x.std(axis=(0, 1, 3), keepdims=True)


def save_model(model, epoch):
    model_save_path = os.path.join(SAV_DIR, f"model_epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


def train_model(model, train_loader, optimizer, criterion):
    total_loss = 0
    for src, tgt in train_loader:
        src = src.view(-1, GIVEN_SEQ, N_CHANNELS)  # Flatten batch and batch size
        tgt = tgt.view(-1, TARGET_SEQ, N_CHANNELS)

        optimizer.zero_grad()
        output = model(src, tgt)
        loss = criterion(output, tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def x_y_split(x):
    X = x[..., :GIVEN_SEQ]
    Y = x[..., GIVEN_SEQ:]
    return X, Y
