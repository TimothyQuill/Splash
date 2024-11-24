import torch
from hyperparameters import *


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
