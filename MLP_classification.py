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
import Dataset_generator_script as dgs
from tqdm import tqdm
import sys

# ---- 3. Training function ----
def train_model(model, data_loader, optimizer, device): 
    model.train()
    total_loss = 0
    total = 0
    Loss_CE = nn.CrossEntropyLoss()

    # FIX 1: Change to unpack a single item, then manually index X and y
    loop = tqdm(data_loader, disable=not sys.stdout.isatty())
    for data in loop:
        
        # We assume the single item 'data' is a list or tuple where the 
        # input tensor (X) is at index 0 and the target tensor (y) is at index 1.
        X, y = data[0], data[1]

        # FIX 2: Move to device AND cast to float32 (Float) to avoid DType mismatch
        X = X.to(device).float() 
        y = y.to(device).float()

        #print(B_idx, D_idx)

        optimizer.zero_grad()
        
        outputs = model(X)

        loss = Loss_CE(outputs, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0) 
        total += X.size(0) 

    avg_loss = total_loss / total
    # For regression, we return 0.0 for accuracy
    return avg_loss, 0.0

# ---- 5. Main training environment ----

def main():

    # Data loading and setup
    data = "Signes_data"
    a = os.listdir(data)
    dataset = dc.CustomDataset(data, a) 

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Safely determine input and output size based on the dataset structure
    # This assumes dataset[0] returns a tuple/list (input_sample, target_sample)

    input_size = len(dataset[0][0])
    output_size = len(dataset) # Assuming target is at index 1
    
    model_ = cvm.Classification_softmax(input_size, output_size).to(device)


    # Define loss and optimizer
    optimizer = optim.Adam(model_.parameters(), lr=1e-2)


    # Training loop

    for epoch in range(1, 21):
        print("Epoch:", epoch)
        # We pass the criterion, as previously discussed
        train_loss, train_acc = train_model(model_, data_loader, optimizer, device) 
        # Only print loss since accuracy is not applicable for MSELoss
        print(f"Epoch {epoch:02d}: Loss={train_loss:.4f}") 

    print("\nTraining complete!")
    torch.save(model_.state_dict(), filename="MLP_interpolation_model.pth")

if __name__ == "__main__":
    main()