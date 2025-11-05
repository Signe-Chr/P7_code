# =============================
#   PyTorch Softmax NN Trainer
# =============================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import Loss_functions as LF
import Cross_validation_models as cvm
import Dataset_class as dc
import os



# ---- 3. Training function ----
def train_model(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()
        outputs = model(X)
        loss = nn.MSELoss(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy


# ---- 4. Evaluation function ----
def evaluate_model(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            loss = nn.MSELoss(outputs, y)#+

            total_loss += loss.item() * X.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy


# ---- 5. Main training environment ----

def main():

    data = "Signes_data"
    a = os.listdir(data)
    dataset = dc.CustomDataset(data, a)


    data_loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)
    Q = [batch for batch in data_loader][0]


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_interpolation = cvm.FilterNet_interpolation(len(Q[0][0]), len(dataset))


    # Define loss and optimizer
    optimizer = optim.Adam(model_interpolation.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(1, 21):
        train_loss, train_acc = train_model(model_interpolation, data_loader, optimizer, device)
        print(f"Epoch {epoch:02d}: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
