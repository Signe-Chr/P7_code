import torch, os
import torch.nn as nn
import torch.optim as optim
import Cross_validation_models as cvm
from tqdm import tqdm
from Test_train_split import load_test_train_data


def train_epoch(model, data, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    X_train = data[0] #Loads all the configurations
    filters_train = data[1]
    n_srcs_train = data[4]
    RIRs_train = data[5]
    indeces_train = data[6] #Loads the indeces for every filter in the dictionary


    #loop = tqdm(data, disable=not sys.stdout.isatty())
    for index in indeces_train:
        X, y = X_train[index].to(torch.float32), index.unsqueeze(0)

        optimizer.zero_grad()
        outputs = model(X).unsqueeze(0) # shape: (batch, num_classes)
        loss = criterion(outputs, y)           # CrossEntropyLoss expects class indices (not one-hot)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    accuracy = 100 * correct / total
    return model, avg_loss, accuracy

# ---------------------------------------------------
# 5. Main training environment
# ---------------------------------------------------
def main():
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_test, data_train, data_val = load_test_train_data()
    input_size = len(data_train[0][0])
    output_size = len(data_train[0])
    epochs = 400
    load = 0

    # Model, loss, optimizer
    model = cvm.FilterNet_classification(input_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-4)
    if os.path.exists(f"MLP_classification_{load}.pth"):
        model.load_state_dict(torch.load(f"MLP_classification_{load}.pth"))
    # Training loop
    for epoch in tqdm(range(1, epochs+1)):
        model, train_loss, train_acc = train_epoch(model, data_train, criterion, optimizer, device)
        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")

    print("\n Training complete!")
    torch.save(model.state_dict(), f"MLP_classification_{epochs+load}.pth")


if __name__ == "__main__":
    main()