import os, sys, torch
import numpy as np
import Cross_validation_models as cvm
#import Dataset_generator_script as dgs
from Loss_functions import MSE, Cosine_similarity, MSEP, AC_loss, compute_H_matrix
from Dataset_class import L, J
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from tqdm import tqdm
import Loss_functions as lf
from Test_train_split import load_test_train_data

def train2(data, wav, epochs, model, dev):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dark_indices = [0,1,2,3,4,5,6,7,8,9,10,11]
    bright_indices = [12]

    alpha=0.5
    beta=0.5
    gamma=0.5
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        

        loop = tqdm(data, desc=f"Epoch {epoch+1}/{epochs}", disable=not sys.stdout.isatty())
        for batch in loop:
            batch_X, flat_batch_y = batch[0].to(dev, dtype=torch.float32), batch[1].to(dev, dtype=torch.float32)
            batch_y = flat_batch_y.reshape(L, J)
            batch_IR = batch[5][0]
            #bright_batch = batch[2][0]
            #dark_batch = batch[3][0]

            optimizer.zero_grad()
            flat_outputs = model(batch_X)
            outputs = flat_outputs.reshape(L, J)
            H = compute_H_matrix(batch_IR)[0].to(dev)
            
            #L_3_reg(q_pred, freqs, L,g_max=1):
            loss = alpha*lf.L_1_reg(outputs, batch_y, H, bright_indices) + (1-alpha)* lf.L_2_reg(outputs, H, dark_indices) + beta*lf.L_3_reg(outputs, L) +  gamma*lf.L_4_reg(outputs, dev)
            #loss = MSE(outputs, batch_y) + Cosine_similarity(flat_outputs, flat_batch_y) + MSEP(outputs, batch_y, batch_IR, batch_IR, wav, bright_batch, dark_batch)[0] + AC_loss(outputs, batch_y, H, bright_batch, dark_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            loop.set_postfix(loss=f"{total_loss/(loop.n+1):.4f}")
            torch.save(model.state_dict(), f"MLP_regression_checkpoint.pth")
    return model

def train(data, wav, epochs, model, dev):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    dark_indices = [0,1,2,3,4,5,6,7,8,9,10,11]
    bright_indices = [12]

    alpha=0.5
    beta=0.5
    gamma=0.5
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

            # Reshape kun hvis størrelsen matcher
            if flat_y.numel() != L*J:
                raise ValueError(f"y-size = {flat_y.numel()}, expected {L*J}")

            y = flat_y.reshape(L, J)

            optimizer.zero_grad()

            out_flat = model(X).unsqueeze(0)
            outputs = out_flat.reshape(L, J)

            H = compute_H_matrix(IR)[0].to(dev)

            loss = alpha*lf.L_1_reg(outputs, y, H, bright_indices) + (1-alpha)* lf.L_2_reg(outputs, H, dark_indices) + beta*lf.L_3_reg(outputs, L) +  gamma*lf.L_4_reg(outputs, dev)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}: loss = {total_loss/len(data):.4f}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_test, X_train = load_test_train_data(test_size=0.25, random_seed=42)
    x_input = torch.tensor([1])
    p_features = len(X_train[0][0])    # Load the first data point and then find the length of the X matrix
    out_features = len(X_train[1][0])  # Length of the flattened filter coeffecients
    compute_cpus = cpu_count()-1
    torch.set_num_threads(compute_cpus)
    #train_loader = DataLoader(X_train, batch_size=1, shuffle=True, num_workers=1)
    #wav = dgs.wav / np.max(np.abs(dgs.wav))
    model = cvm.FilterNet_regression(p_features, out_features).to(device)
    if os.path.exists("MLP_regression_checkpoint.pth"):
        model.load_state_dict(torch.load("MLP_regression_checkpoint.pth"))
        print("Succesfully loaded the latest checkpoint of the model")
    elif os.path.exists("MLP_regression.pth"):
        model.load_state_dict(torch.load("MLP_regression.pth"))
        print("Succesfully loaded a previously trained model")
    model = train(X_train, x_input.to(device), epochs=15, dev=device, model=model)
    torch.save(model.state_dict(), f"MLP_regression.pth")