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

# ---- 3. Training function ----
def train_model(model, data_loader, optimizer, device, wav): 
    model.train()
    total_loss = 0
    total = 0

    # FIX 1: Change to unpack a single item, then manually index X and y
    for data in data_loader:
        
        # We assume the single item 'data' is a list or tuple where the 
        # input tensor (X) is at index 0 and the target tensor (y) is at index 1.
        X, y = data[0], data[1]

        # FIX 2: Move to device AND cast to float32 (Float) to avoid DType mismatch
        X = X.to(device).float() 
        y = y.to(device).float()
        rir = data[5].to(device)
        B_idx = data[2].to(device)
        D_idx = data[3].to(device)

        

        optimizer.zero_grad()
        
        outputs = model(X)

        H = LF.compute_H_matrix(rir, fs=16000, n_fft=None)
        loss = nn.MSELoss(outputs, y) + LF.L_2_loss(outputs, y)+ LF.L_3_loss(outputs, y, rir, rir, wav, B_idx) + LF.L_4_loss(outputs, rir, wav, H, B_idx, D_idx)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0) 
        total += X.size(0) 

    avg_loss = total_loss / total
    # For regression, we return 0.0 for accuracy
    return avg_loss, 0.0


# ---- 4. Evaluation function ----
def evaluate_model(model, dataloader, device, criterion): 
    model.eval()
    total_loss = 0
    total = 0 

    with torch.no_grad():
        for data in dataloader:
            # FIX: Manually unpack X and y
            X, y = data[0], data[1]

            # FIX: Move to device AND cast to float32 (Float)
            X = X.to(device).float()
            y = y.to(device).float()
            
            outputs = model(X)
            loss = criterion(outputs, y) 

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

    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Safely determine input and output size based on the dataset structure
    # This assumes dataset[0] returns a tuple/list (input_sample, target_sample)
    if isinstance(dataset[0], (list, tuple)):
        input_size = len(dataset[0][0]) 
        output_size = len(dataset[0][1]) # Assuming target is at index 1
    else:
        # Fallback if the dataset returns a single concatenated tensor (less common)
        print("Warning: Dataset structure is ambiguous. Assuming input/output size from the first sample.")
        input_size = len(dataset[0])
        output_size = len(dataset[0]) 

    
    model_interpolation = cvm.FilterNet_interpolation(input_size, output_size).to(device)


    # Define loss and optimizer
    optimizer = optim.Adam(model_interpolation.parameters(), lr=1e-3)
    #criterion = nn.MSELoss()
    wav = dgs.wav

    # Training loop
    for epoch in range(1, 21):
        # We pass the criterion, as previously discussed
        train_loss, train_acc = train_model(model_interpolation, data_loader, optimizer, device, wav) 
        # Only print loss since accuracy is not applicable for MSELoss
        print(f"Epoch {epoch:02d}: Loss={train_loss:.4f}") 

    print("\nTraining complete!")


if __name__ == "__main__":
    main()