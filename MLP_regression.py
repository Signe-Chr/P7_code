import os, sys, torch
import numpy as np
import Cross_validation_models as cvm
import Dataset_generator_script as dgs
from Loss_functions import MSE, Cosine_similarity, MSEP, AC_loss, compute_H_matrix
import Loss_functions as lf
from Dataset_class import CustomDataset, L, J
from torch.utils.data import DataLoader
from multiprocessing import cpu_count
from tqdm import tqdm

def train(data, wav, epochs, model, dev):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        loop = tqdm(data, desc=f"Epoch {epoch+1}/{epochs}", disable=not sys.stdout.isatty())
        for batch in loop:
            batch_X, flat_batch_y = batch[0].to(dev, dtype=torch.float32), batch[1].to(dev, dtype=torch.float32)
            batch_y = flat_batch_y.reshape(L, J)
            batch_IR = batch[5][0]
            bright_batch = batch[2][0]
            dark_batch = batch[3][0]

            optimizer.zero_grad()
            flat_outputs = model(batch_X)
            outputs = flat_outputs.reshape(L, J)
            H = compute_H_matrix(batch_IR)[0].to(dev)
            dark_indices = [0,1,2,3,4,5,6,7,8,9,10,11]
            bright_indices = [12]


            alpha=0.5
            beta=0.5
            gamma=0.5
            #L_3_reg(q_pred, freqs, L,g_max=1):
            loss = alpha*lf.L_1_reg(outputs, batch_y, H, bright_indices) + (1-alpha)* lf.L_2_reg(outputs, H, dark_indices) + beta*lf.L_3_reg(outputs, L) +  gamma*lf.L_4_reg(outputs, dev)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            loop.set_postfix(loss=f"{total_loss/(loop.n+1):.4f}")
            torch.save(model.state_dict(), f"MLP_regression_checkpoint.pth")
    return model

if __name__ == "__main__":

    widths = range(2, 11)
    depths = range(2, 11)
    heights = range(4, 13)
    rooms = [[width, depth, height/2] for width in widths for depth in depths for height in heights]+[[100, 100, 100]]  # if width<=depth (This code snippet can be added in the list loop if rooms like 2x10xz and 10x2xz are deemed equal)
    target_rooms = [
    [2, 2, 2], [2, 4, 3], [ 2, 6, 4], [ 2,  8, 3], [ 2, 10, 3],
    [4, 2, 3], [4, 4, 2], [ 4, 8, 4], [ 4, 10, 4], [ 6,  2, 2],
    [6, 4, 3], [6, 6, 3], [ 6, 8, 4], [ 6, 10, 4], [ 8,  2, 3],
    [8, 6, 3], [8, 8, 4], [10, 2, 3], [10,  6, 5], [10, 10, 4]
    ]
    room_indices = [i for i in range(len(rooms)) if rooms[i] in target_rooms]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_dir = "Signes_data"
    full_data = os.listdir(data_dir)
    train_points = []
    for data in full_data:
        i = int(data.split("_")[1])
        if (i in room_indices) and (i not in room_indices[::4]):
            train_points.append(data)
    trainset = CustomDataset(data_dir, train_points)
    p_features = len(trainset[0][0])    # Load the first data point and then find the length of the X matrix
    out_features = len(trainset[0][1])  # Length of the flattened filter coeffecients
    compute_cpus = cpu_count()-1
    torch.set_num_threads(compute_cpus)
    train_loader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=1)

    
    wav = np.array([1])#dgs.wav / np.max(np.abs(dgs.wav))
    model = cvm.FilterNet_regression(p_features, out_features).to(device)
    #if os.path.exists("MLP_regression_checkpoint.pth"):
    #    model.load_state_dict(torch.load("MLP_regression_checkpoint.pth"))
    #    print("Succesfully loaded the latest checkpoint of the model")
    #elif os.path.exists("MLP_regression.pth"):
    #    model.load_state_dict(torch.load("MLP_regression.pth"))
    #    print("Succesfully loaded a previously trained model")
    model = train(train_loader, torch.from_numpy(wav).to(device), epochs=15, dev=device, model=model)
    torch.save(model.state_dict(), f"MLP_regression.pth")