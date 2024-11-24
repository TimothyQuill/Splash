def train_model():
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


def evaluate_model():
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


import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import train_test_split
from model import SequenceTransformer
from data import generate_dataset
from plot import plot_loss
from hyperparameters import *
from helper import *



# Data preparation
x = torch.Tensor(generate_dataset())

# Normalise the data
x = normalise(x)

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
    train_loss = train_model()
    test_loss = evaluate_model()
    print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
    train_losses.append(train_loss)
    test_losses.append(test_loss)

    plot_loss(train_losses, test_losses)

    # Save the model checkpoint
    save_model(model, epoch)
