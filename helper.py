import os
import torch

from torch.utils.data import DataLoader, TensorDataset
from hyperparameters import *


def create_data_loader(x, y):
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


def create_dir():
    os.makedirs(SAV_DIR, exist_ok=True)


def load_model(model_class, save_path, d_model=64):
    model = model_class(d_model=d_model)
    model.load_state_dict(torch.load(save_path))
    model.eval()  # Set to evaluation mode
    print(f"Model loaded from {save_path}")
    return model


def normalise(x):
    return (x - x.mean(axis=(0, 1, 3), keepdims=True)) / x.std(axis=(0, 1, 3), keepdims=True)


def save_model(model, epoch):
    model_save_path = os.path.join(SAV_DIR, f"model_epoch_{epoch + 1}.pth")
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved to {model_save_path}")


def x_y_split(x):
    X = x[..., :GIVEN_SEQ]
    Y = x[..., GIVEN_SEQ:]
    return X, Y
