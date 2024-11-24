"""
High-level training process.
"""
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from model import SequenceTransformer
from data import generate_dataset
from plot import plot_loss
from helper import *


# Generate the data
x = torch.Tensor(generate_dataset())

# Scale the data to unit range
x = minmax_scaling(x)

# Split into X and Y
X, Y = x_y_split(x)

# Split into train and test set
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE)

# Create data loaders
train_loader = create_data_loader(X_train, Y_train)
test_loader = create_data_loader(X_test, Y_test)

# Define the model
model = SequenceTransformer()

# Training setup
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Ensure the save directory exists
create_dir()

train_losses = []
test_losses = []

# Training loop
for epoch in range(EPOCHS):
    train_loss = train_model(model, train_loader, optimizer, criterion)
    test_loss = evaluate_model(model, test_loader, criterion)
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    plot_loss(train_losses, test_losses)
    save_model(model, epoch)
