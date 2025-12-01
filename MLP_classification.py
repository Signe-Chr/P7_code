import torch, sys, os
import torch.nn as nn
import torch.optim as optim
import Loss_functions as LF
import Cross_validation_models as cvm
import Dataset_class as dc
import Dataset_generator_script as dgs
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from Test_train_split import load_test_train_data

"""
def noget():
    # Unpack data
    a1, a2, a3, a4, a5, a6, a7, a8 = data
    # Repack data to make sure it has the lengths it's supposed to
    data = [a1, a2, np.array([a3]).T, np.array([a4]).T, a5, a6, a7, a8]
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        # Use zip(*data) to be able to loop over singular datapoints
        for sample in zip(*data):   # <-- ét datapunkt ad gangen

            # Pak datapunktet ud
            X = torch.tensor(sample[0], dtype=torch.float32).to(dev)
            flat_y = torch.tensor(sample[1], dtype=torch.float32).to(dev).unsqueeze(0)
            IR = sample[5]    # eller sample[2], afhængigt af strukturen
"""

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
        X, y = X_train[index], index

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
#
    #from Dataset_generator_script import room_indices as ri
    ## Data loading and setup
    #data_dir = "Signes_data"
    #a = os.listdir(data_dir)
#
#
    #train_set = []
    #for data in a:
    #    r = int(data.split("_")[1])
    #    if (r not in ri[::4]):
    #        train_set.append(data)
#
#
    #dataset = dc.CustomDataset(data_dir, train_set)
    #input_size = len(dataset[0][0])
    #output_size = len(dataset)
#
    #data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    data_test, data_train = load_test_train_data()

    x_input = torch.tensor([1])
#
    input_size = len(data_train[0][0])
    output_size = len(data_train)

    # Model, loss, optimizer
    model = cvm.FilterNet_classification(input_size, output_size).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    if os.path.exists("MLP_classification.pth"):
        model.load_state_dict(torch.load("MLP_classification.pth"))
    # Training loop
    for epoch in range(1, 41):
        train_loss, train_acc = train_epoch(model, data_train, criterion, optimizer, device)
        print(f"Epoch {epoch:02d} | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")

    print("\n Training complete!")
    torch.save(model.state_dict(), "MLP_classification.pth")


if __name__ == "__main__":
    main()