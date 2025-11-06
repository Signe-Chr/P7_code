import os
import sys
import torch
import numpy as np
import Cross_validation_models as cvm
import Dataset_generator_script as dgs
from Loss_functions import Cosine_similarity, MSEP, AC_loss, compute_H_matrix
from Dataset_class import CustomDataset, L, J
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from tqdm import tqdm

def train(data, wav, epochs, model, dev):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    MSE_loss = torch.nn.MSELoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        loop = tqdm(data, desc=f"Epoch {epoch+1}/{epochs}", disable=not sys.stdout.isatty())
        for batch in loop:
            batch_X, pre_flat_batch_y = batch[0].to(dev, dtype=torch.float32), batch[1].to(dev, dtype=torch.float32)
            batch_y = pre_flat_batch_y.reshape(L, J)
            batch_IR = batch[5][0]
            bright_batch = batch[2][0]
            dark_batch = batch[3][0]

            optimizer.zero_grad()
            pre_flat_outputs = model(batch_X)
            outputs = pre_flat_outputs.reshape(L, J)
            H = compute_H_matrix(batch_IR)[0].to(dev)

            loss = MSE_loss(outputs, batch_y) + Cosine_similarity(pre_flat_outputs, pre_flat_batch_y) + MSEP(outputs, batch_y, batch_IR, batch_IR, wav, bright_batch) + AC_loss(outputs, batch_y, H, bright_batch, dark_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            loop.set_postfix(loss=f"{total_loss/(loop.n+1):.4f}")
            torch.save(model.state_dict(), f"MLP_regression_checkpoint.pth")
    return model

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "VAST_filter_archive_730"
    from Dataset_generator_script import room_indices as ri
    full_data = os.listdir(data_dir)
    train_points = []
    for data in full_data:
        i = int(data.split("_")[1])
        if (i in ri) and (i not in ri[::4]):
            train_points.append(data)
    trainset = CustomDataset(data_dir, train_points)
    p_features = len(trainset[0][0])    # Load the first data point and then find the length of the X matrix
    out_features = len(trainset[0][1])  # Length of the flattened filter coeffecients
    compute_cpus = cpu_count()-1
    torch.set_num_threads(compute_cpus)
    train_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=1)
    wav = dgs.wav / np.max(np.abs(dgs.wav))
    model = cvm.model_.to(device)
    if os.path.exists("MLP_regression_checkpoint.pth"):
        model.load_state_dict(torch.load("MLP_regression_checkpoint.pth"))
        print("Succesfully loaded the latest checkpoint of the model")
    elif os.path.exists("MLP_regression.pth"):
        model.load_state_dict(torch.load("MLP_regression.pth"))
        print("Succesfully loaded a previously trained model")
    model = train(train_loader, torch.from_numpy(wav).to(device), epochs=15, dev=device, model=model)
    torch.save(model.state_dict(), f"MLP_regression.pth")