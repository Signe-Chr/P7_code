import torch, sys, os
import torch.nn as nn
import torch.optim as optim
import Loss_functions as LF
import Cross_validation_models as cvm
import Dataset_class as dc
import Dataset_generator_script as dgs
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(dataloader, disable=not sys.stdout.isatty())
    for data in loop:
        X, y = data[0].to(device).float(), data[6].to(device)

        optimizer.zero_grad()
        outputs = model(X)                     # shape: (batch, num_classes)
        loss = criterion(outputs, y)           # CrossEntropyLoss expects class indices (not one-hot)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# ---------------------------------------------------
# 5. Main training environment
# ---------------------------------------------------
def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from Dataset_generator_script import room_indices as ri
    # Data loading and setup
    data_dir = "Signes_data"
    a = os.listdir(data_dir)


    train_set = []
    for data in a:
        r = int(data.split("_")[1])
        if (r not in ri[::4]):
            train_set.append(data)


    dataset = dc.CustomDataset(data_dir, train_set)
    

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    input_size = len(dataset[0][0])
    output_size = len(dataset)

    # Model, loss, optimizer
    model = cvm.FilterNet_classification(input_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    if os.path.exists("MLP_classification.pth"):
        model.load_state_dict(torch.load("MLP_classification.pth"))
    # Training loop
    for epoch in range(1, 41):
        train_loss, train_acc = train_epoch(model, data_loader, criterion, optimizer, device)
        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")

    print("\n Training complete!")
    torch.save(model.state_dict(), "MLP_classification.pth")


if __name__ == "__main__":
    main()