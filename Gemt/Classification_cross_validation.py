import torch
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
#from Baseline_fast_search import ANN_Search_and_Refine
from scipy.io import wavfile
from Dataset_class import CustomDataset, L, J
from torch.utils.data import DataLoader
from Gemt.Dataset_generator_script import room_indices as ri
import os

#---Check Device--
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#---Define some variables---
num_layers=3
num_folds=5
num_epochs=200
fcentres = torch.tensor([1000, 2000], device=device)

#---Load data---
data_dir="Signes_data"
full_data=os.listdir(data_dir)
data_points = []
for data in full_data:
    if int(data.split("_")[1]) in ri:
        data_points.append(data)
data=CustomDataset(data_dir,data_points)
data_loader = DataLoader(data, batch_size=len(data), shuffle=True)
Q=[batch for batch in data_loader][0]
X=Q[0]
filters=Q[1]
bright_zone_mics_index=Q[2]
dark_zone_mics_index=Q[3]
n_srcs=Q[4]
IR=Q[5]
print(X.shape)

#---Define more variables---
M_B = len(bright_zone_mics_index)
M_D = len(dark_zone_mics_index)
L=n_srcs[0].item()

#--- Load sound file---
wav_path = "relaxing-guitar-loop-v5-245859.wav"
fs_wav, wav = wavfile.read(wav_path)
if wav.ndim > 1:
    wav = np.mean(wav, axis=1)
wav = wav[5*fs_wav : 7*fs_wav]
wav = wav / np.max(np.abs(wav))  # scale to [-1,1]
x_input = torch.from_numpy(wav.astype(np.float32)).unsqueeze(0)
x_input = x_input.to(device)

num_total, input_size = X.shape
filter_length = filters.shape[1]

configs_tensor = X       # [N_total, num_features]
filters_tensor = filters  # [N_total, filter_length]
# Encode unique filters as integers
unique_filters, y_indices = np.unique(filters, axis=0, return_inverse=True)
y_tensor = torch.from_numpy(y_indices).long()  # [N_total]
num_classes = len(unique_filters)

print(f"Configs shape: {configs_tensor.shape}")
print(f"Filters shape: {filters_tensor.shape}") 
print(f"Number of classes: {num_classes}")
#---One hidden layer---
if num_layers == 1:
    neurons = [128, 256, 512]
    train_acc_list=[]
    test_acc_list=[]
    for neuron in neurons:
        fold_train_acc = []
        fold_test_acc = []

        for folds in range(number_of_folds):
            # --- Split data for cross-validation ---
            X_train, X_test, y_train, y_test = train_test_split(
            configs_tensor, y_tensor, test_size=0.25,shuffle=True,random_state=folds)
            # --- Define model
            model = torch.nn.Sequential(
                torch.nn.Linear(input_size, neuron),
                torch.nn.ReLU(),
                torch.nn.Linear(neuron, num_classes)
            )
            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Optional: print model summary
            summary(model, input_size=(input_size,))

            # --- Training Loop ---
            model.train()
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, y_train)
                loss.backward()
                optimizer.step()

            # --- Evaluation ---
            model.eval()
            with torch.no_grad():
                train_pred_logits = model(X_train)
                test_pred_logits = model(X_test)
                
                pred_classes_train = train_pred_logits.argmax(dim=1)
                pred_classes_test = test_pred_logits.argmax(dim=1)

                true_filters_train = unique_filters[y_train.numpy()]
                true_filters_test = unique_filters[y_test.numpy()]
                
                pred_filters_train = unique_filters[pred_classes_train.numpy()]
                pred_filters_test = unique_filters[pred_classes_test.numpy()]
                
                train_acc=np.mean((true_filters_train-pred_filters_train) ** 2).item()
                test_acc=np.mean((true_filters_test-pred_filters_test) ** 2).item()

                fold_train_acc.append(train_acc)
                fold_test_acc.append(test_acc)
            print(f"Neuron: {neuron}, Fold {folds+1} -- Train MSE: {train_acc:.4f}, Test MSE: {test_acc:.4f}")
        
        print(f"Neuron: {neuron} -- Average Train Acc: {np.mean(fold_train_acc):.4f}, "
          f"Average Test Acc: {np.mean(fold_test_acc):.4f}\n")
        train_acc_list.append(np.mean(fold_train_acc))
        test_acc_list.append(np.mean(fold_train_acc))
    for neuronss,train,test in zip(neurons,train_acc_list,test_acc_list):
        print(f"Average over 5 folds: (Neurons, Train MSE, Test MSE): ({neuronss}, {train:.20f}, {test:.20f})")
        
        
        
#---Two hidden layers---
if num_layers == 2:
    neurons1 = [128, 256, 512]
    neurons2 = [128, 256, 512]
    test_mse_grid = np.zeros((len(neurons1), len(neurons2)))
    train_mse_grid = np.zeros((len(neurons1), len(neurons2)))

    for i, neuron1 in enumerate(neurons1):
        for j, neuron2 in enumerate(neurons2):

            # Cross-validation over folds
            train_mse_folds = []
            test_mse_folds = []

            for folds in range(number_of_folds):
                # --- Split data ---
                X_train, X_test, y_train, y_test = train_test_split(
                    configs_tensor, y_tensor, test_size=0.25, shuffle=True, random_state=folds
                )

                # --- Define model ---
                model = torch.nn.Sequential(
                    torch.nn.Linear(input_size, neuron1),
                    torch.nn.ReLU(),
                    torch.nn.Linear(neuron1, neuron2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(neuron2, num_classes)
                )
                criterion = torch.nn.CrossEntropyLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                # --- Training Loop ---
                model.train()
                for epoch in range(num_epochs):
                    optimizer.zero_grad()
                    outputs = model(X_train)
                    loss = criterion(outputs, y_train)
                    loss.backward()
                    optimizer.step()

                # --- Evaluation ---
                model.eval()
                with torch.no_grad():
                    train_pred_logits = model(X_train)
                    test_pred_logits = model(X_test)

                    pred_classes_train = train_pred_logits.argmax(dim=1)
                    pred_classes_test = test_pred_logits.argmax(dim=1)

                    true_filters_train = unique_filters[y_train.numpy()]
                    true_filters_test = unique_filters[y_test.numpy()]
                    pred_filters_train = unique_filters[pred_classes_train.numpy()]
                    pred_filters_test = unique_filters[pred_classes_test.numpy()]

                    # Mean squared error between predicted and true filters
                    train_mse = np.mean((true_filters_train - pred_filters_train) ** 2)
                    test_mse = np.mean((true_filters_test - pred_filters_test) ** 2)

                    train_mse_folds.append(train_mse)
                    test_mse_folds.append(test_mse)

            # Average across folds
            train_mse_grid[i, j] = np.mean(train_mse_folds)
            test_mse_grid[i, j] = np.mean(test_mse_folds)

    # --- Plot heatmaps ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def plot_mse_grid(ax, mse_grid, title):
        im = ax.imshow(mse_grid, origin='lower', cmap='viridis')
        for x in range(mse_grid.shape[0]):
            for y in range(mse_grid.shape[1]):
                ax.text(y, x, f"{mse_grid[x, y]:.4f}", ha='center', va='center', color='w')
        ax.set_xticks(np.arange(len(neurons2)))
        ax.set_xticklabels(neurons2)
        ax.set_yticks(np.arange(len(neurons1)))
        ax.set_yticklabels(neurons1)
        ax.set_xlabel("Neurons in 2nd layer")
        ax.set_ylabel("Neurons in 1st layer")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    plot_mse_grid(axes[0], train_mse_grid, "Train MSE")
    plot_mse_grid(axes[1], test_mse_grid, "Test MSE")
    plt.tight_layout()
    plt.show()
    
if num_layers == 3:
    neurons1 = [128, 256, 512]
    neurons2 = [128, 256, 512]
    neurons3 = [128, 256, 512]

    all_test_acoustic_grids = []
    all_train_acoustic_grids = []

    for k, neuron3 in enumerate(neurons3):
        print(f'Neurons in 3rd layer {neuron3}')
        test_acoustic_grid = np.zeros((len(neurons2), len(neurons1)))
        train_acoustic_grid = np.zeros((len(neurons2), len(neurons1)))

        for i, neuron2 in enumerate(neurons2):
            print(f'Neurons in 2nd layer {neuron2}')
            for j, neuron1 in enumerate(neurons1):
                print(f'Neurons in 1st layer {neuron1}')

                fold_train_acoustic = []
                fold_test_acoustic = []

                # Cross-validation
                kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

                for fold_idx, (train_idx, test_idx) in enumerate(kf.split(configs_tensor)):
                    print(f"\nFold {fold_idx+1}")
                    print(f"Train samples: {len(train_idx)}, Test samples: {len(test_idx)}")

                    # Convert indices to tensors and select corresponding data
                    X_train, X_test = configs_tensor[train_idx], configs_tensor[test_idx]
                    y_train, y_test = y_tensor[train_idx], y_tensor[test_idx]
                    IR_train, IR_test = IR[train_idx], IR[test_idx]

                    # --- Define model ---
                    model = torch.nn.Sequential(
                        torch.nn.Linear(input_size, neuron1),
                        torch.nn.ReLU(),
                        torch.nn.Linear(neuron1, neuron2),
                        torch.nn.ReLU(),
                        torch.nn.Linear(neuron2, neuron3),
                        torch.nn.ReLU(),
                        torch.nn.Linear(neuron3, num_classes)
                    ).to(device)

                    criterion = torch.nn.CrossEntropyLoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                    # --- Training loop ---
                    model.train()
                    for epoch in range(num_epochs):
                        optimizer.zero_grad()
                        logits = model(X_train)
                        loss = criterion(logits, y_train)
                        loss.backward()
                        optimizer.step()

                    # --- Evaluation ---
                    model.eval()
                    with torch.no_grad():
                        # Predict class indices
                        train_logits = model(X_train)
                        test_logits = model(X_test)
                        pred_classes_train = train_logits.argmax(dim=1)
                        pred_classes_test = test_logits.argmax(dim=1)

                        # Map class indices to filter vectors
                        pred_filters_train = unique_filters[pred_classes_train.cpu().numpy()]
                        pred_filters_test = unique_filters[pred_classes_test.cpu().numpy()]
                        true_filters_train = unique_filters[y_train.cpu().numpy()]
                        true_filters_test = unique_filters[y_test.cpu().numpy()]

                        # --- Compute acoustic loss via ANN_Search_and_Refine ---
                        # Train loss
                        train_loss, _ = ANN_Search_and_Refine(
                            test_filters=torch.tensor(pred_filters_train, dtype=torch.float32, device=device),
                            dictionary=torch.tensor(true_filters_train, dtype=torch.float32, device=device),
                            IR_train=IR_train_tensor,
                            IR_test=IR_train_tensor,  # Evaluate on same train RIRs
                            fcentres=fcentres,
                            M_B=M_B,
                            M_D=M_D,
                            x_input=x_input,
                            k_neighbors=20
                        )

                        # Test loss
                        test_loss, _ = ANN_Search_and_Refine(
                            test_filters=torch.tensor(pred_filters_test, dtype=torch.float32, device=device),
                            dictionary=torch.tensor(true_filters_train, dtype=torch.float32, device=device),
                            IR_train=IR_train_tensor,
                            IR_test=IR_test_tensor,
                            fcentres=fcentres,
                            M_B=M_B,
                            M_D=M_D,
                            x_input=x_input,
                            k_neighbors=20
                        )

                        fold_train_acoustic.append(train_loss)
                        fold_test_acoustic.append(test_loss)

                # Average over folds
                train_acoustic_grid[i, j] = np.mean(fold_train_acoustic)
                test_acoustic_grid[i, j] = np.mean(fold_test_acoustic)

        all_train_acoustic_grids.append(train_acoustic_grid)
        all_test_acoustic_grids.append(test_acoustic_grid)

    # --- Plotting ---
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(2, len(neurons3), figure=fig, wspace=0.4, hspace=0.4)

    vmin = min([g.min() for g in all_train_acoustic_grids + all_test_acoustic_grids])
    vmax = max([g.max() for g in all_train_acoustic_grids + all_test_acoustic_grids])

    for k, neuron3 in enumerate(neurons3):
        # Test acoustic
        ax = fig.add_subplot(gs[0, k])
        im_test = ax.imshow(all_test_acoustic_grids[k], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        for x in range(len(neurons2)):
            for y in range(len(neurons1)):
                ax.text(y, x, f"{all_test_acoustic_grids[k][x, y]:.4f}", ha='center', va='center', color='w', fontsize=8)
        ax.set_xticks(range(len(neurons1)))
        ax.set_xticklabels(neurons1)
        ax.set_yticks(range(len(neurons2)))
        ax.set_yticklabels(neurons2)
        ax.set_xlabel("L1 neurons")
        ax.set_ylabel("L2 neurons")
        ax.set_title(f"Test Acoustic Loss, L3={neuron3}")

        # Train acoustic
        ax = fig.add_subplot(gs[1, k])
        im_train = ax.imshow(all_train_acoustic_grids[k], origin='lower', cmap='viridis', vmin=vmin, vmax=vmax)
        for x in range(len(neurons2)):
            for y in range(len(neurons1)):
                ax.text(y, x, f"{all_train_acoustic_grids[k][x, y]:.4f}", ha='center', va='center', color='w', fontsize=8)
        ax.set_xticks(range(len(neurons1)))
        ax.set_xticklabels(neurons1)
        ax.set_yticks(range(len(neurons2)))
        ax.set_yticklabels(neurons2)
        ax.set_xlabel("L1 neurons")
        ax.set_ylabel("L2 neurons")
        ax.set_title(f"Train Acoustic Loss, L3={neuron3}")

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im_test, cax=cbar_ax, label='Acoustic Loss')

    plt.show()




    