import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from model import SequenceTransformer
from data import generate_dataset
from hyperparameters import *
from training import train_model, evaluate_model
from plot import plot_loss


# Data preparation
x = torch.Tensor(generate_dataset())

# Normalise the data
x = (x - x.mean(axis=(0, 1, 3), keepdims=True)) / x.std(axis=(0, 1, 3), keepdims=True)

X = x[..., :4]  # First 4 numbers
Y = x[..., 4:]  # Remaining 26 numbers

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

dataset = TensorDataset(X_train, Y_train)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

dataset = TensorDataset(X_test, Y_test)
test_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = SequenceTransformer()

# Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

train_losses = []
test_losses = []
for epoch in range(EPOCHS):
    train_loss = train_model(model, train_loader, optimizer, criterion)
    test_loss = evaluate_model(model, test_loader, criterion)
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    plot_loss(train_losses, test_losses)
