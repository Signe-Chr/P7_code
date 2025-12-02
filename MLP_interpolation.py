# =============================
#   Interpolation MLP
# =============================
import torch, torch.optim as optim
import Loss_functions as LF
import Cross_validation_models as cvm
import Dataset_class as dc
from tqdm import tqdm
from Test_train_split import load_test_train_data

# ---- 1. Training function ----
def train_model(model, data, optimizer, device, wav): 
    model.train()
    total_loss = 0
    total = 0
    X_train = data[0] #All configurations
    filters_train = data[1] #All filters
    RIRs_train = data[5] #All RIRs
    indeces_train = data[6] #Loads the indeces for every filter in the dictionary

    for index in tqdm(indeces_train):
        X, y = X_train[index].to(torch.float32), filters_train[index]
        X = X.to(device).float() 
        y = y.to(device).float().unsqueeze(0)
        rir = RIRs_train[index].to(device)
        B_idx = torch.tensor([12]).to(device)
        D_idx = torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11]).to(device)

        optimizer.zero_grad()
        coefficients = model(X)
        outputs = torch.matmul(filters_train.T.float() , coefficients.T.float()).T.unsqueeze(0)

        H = LF.compute_H_matrix(rir, fs=16000, n_fft=None)[0].to(device)
        loss = 1/4 * (LF.MSE(outputs, y) + LF.Cosine_similarity(outputs, y) + LF.MSEP(outputs.reshape(dc.L,dc.J), y.reshape(dc.L,dc.J), rir, wav, B_idx, D_idx)[0] + LF.AC_loss(outputs.reshape(dc.L,dc.J), y.reshape(dc.L,dc.J), H, B_idx, D_idx))
        #print(LF.MSE(outputs, y), LF.Cosine_similarity(outputs, y), LF.MSEP(outputs.reshape(dc.L,dc.J), y.reshape(dc.L,dc.J), rir, wav, B_idx, D_idx)[0], LF.AC_loss(outputs.reshape(dc.L,dc.J), y.reshape(dc.L,dc.J), H, B_idx, D_idx))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0) 
        total += X.size(0)

    avg_loss = total_loss / total
    return avg_loss


# ---- 2. Main training environment ----
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_test, data_train = load_test_train_data()
    input_size = len(data_train[0][0])
    output_size = len(data_train[0]) # Assuming target is at index 1

    model_interpolation = cvm.FilterNet_interpolation(input_size, output_size).to(device)
    optimizer = optim.Adam(model_interpolation.parameters(), lr=1e-2)
    x_input = torch.tensor([1])

    # Training loop
    for epoch in range(1,51):
        print("Epoch:", epoch)
        train_loss = train_model(model_interpolation, data_train, optimizer, device, x_input) 
        print(f"Epoch {epoch:02d}: Loss={train_loss:.4f}")
        torch.save(model_interpolation.state_dict(), "MLP_interpolation_checkpoint.pth")
        

    print("\nTraining complete!")
    torch.save(model_interpolation.state_dict(), "MLP_interpolation.pth")

if __name__ == "__main__":
    main()