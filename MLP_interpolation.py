# =============================
#   PyTorch Softmax NN Trainer
# =============================
import torch, os
import torch.nn as nn
import torch.optim as optim
import numpy as np
import Loss_functions as LF
import Cross_validation_models as cvm
import Dataset_class as dc
import Dataset_generator_script as dgs
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from Train_test_split import load_test_train_data
from Dataset_generator_script import room_indices as ri

# ---- 3. Training function ----
def train_model(model, data_loader, optimizer, device, wav, YY): 
    model.train()
    total_loss = 0
    total = 0

    # FIX 1: Change to unpack a single item, then manually index X and y
    loop = tqdm(data_loader)#, disable=not sys.stdout.isatty())
    for data in loop:
        
        # We assume the single item 'data' is a list or tuple where the 
        # input tensor (X) is at index 0 and the target tensor (y) is at index 1.
        X, y = data[0], data[1]

        # FIX 2: Move to device AND cast to float32 (Float) to avoid DType mismatch
        X = X.to(device).float() 
        y = y.to(device).float()
        rir = data[5][0].to(device)
        B_idx = torch.tensor(data[2]).to(device)
        D_idx = torch.tensor(data[3]).to(device)
        #print(B_idx, D_idx)

        optimizer.zero_grad()
        
        coefficients = model(X)

        #print(coefficients)
        #coefficients = torch.softmax(coefficients, 1)
        outputs = torch.matmul(YY.T.float() , coefficients.T.float()).T

        H = LF.compute_H_matrix(rir, fs=16000, n_fft=None)[0].to(device)

        #print(H.shape)
        loss = 0.000001*(LF.MSE(outputs, y) + LF.Cosine_similarity(outputs, y) + LF.MSEP(outputs.reshape(dc.L,dc.J), y.reshape(dc.L,dc.J), rir, wav, B_idx, D_idx)[0] + LF.AC_loss(outputs.reshape(dc.L,dc.J), y.reshape(dc.L,dc.J), H, B_idx, D_idx))
        #print(YY, coefficients)
        #print(LF.MSE(outputs, y), LF.Cosine_similarity(outputs, y), LF.MSEP(outputs.reshape(dc.L,dc.J), y.reshape(dc.L,dc.J), rir, wav, B_idx, D_idx)[0], LF.AC_loss(outputs.reshape(dc.L,dc.J), y.reshape(dc.L,dc.J), H, B_idx, D_idx))
        loss.backward()
        optimizer.step()
        

        total_loss += loss.item() * X.size(0) 
        total += X.size(0)
        loop.set_postfix(loss=f"{total_loss/(loop.n+1):.4f}")

    avg_loss = total_loss / total
    # For regression, we return 0.0 for accuracy
    return avg_loss, 0.0

# ---- 5. Main training environment ----

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Data loading and setup
    data_dir = "Signes_data"
    a = os.listdir(data_dir)


    train_set = []
    for data in a:
        r = int(data.split("_")[1])
        if (r not in ri[::4]):
            train_set.append(data)
    X_train, X_test = load_test_train_data(test_size=0.25, random_seed=42) # Finde ud af hvordan man bruger denne istedet for det andet bøvl


    dataset = dc.CustomDataset(data_dir, train_set)

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    # Safely determine input and output size based on the dataset structure
    # This assumes dataset[0] returns a tuple/list (input_sample, target_sample)

    input_size = len(dataset[0][0])
    output_size = len(dataset) # Assuming target is at index 1
    
    data_loader_y = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    YY = [batch for batch in data_loader_y][0][1].to(device)

    
    model_interpolation = cvm.FilterNet_interpolation(input_size, output_size).to(device)


    # Define loss and optimizer
    optimizer = optim.Adam(model_interpolation.parameters(), lr=1e-2)
    #criterion = nn.MSELoss()
    wav = torch.from_numpy(dgs.wav).to(device)

    # Training loop
    for epoch in range(1, 21):
        print("Epoch:", epoch)
        # We pass the criterion, as previously discussed
        train_loss, train_acc = train_model(model_interpolation, data_loader, optimizer, device, wav, YY) 
        # Only print loss since accuracy is not applicable for MSELoss
        print(f"Epoch {epoch:02d}: Loss={train_loss:.4f}")
        torch.save(model_interpolation.state_dict(), "MLP_interpolation_checkpoint.pth")
        

    print("\nTraining complete!")
    torch.save(model_interpolation.state_dict(), "MLP_interpolation.pth")

if __name__ == "__main__":
    main()