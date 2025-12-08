import torch.nn as nn
from Test_train_split import load_test_train_data
import torch.optim as optim
import torch
import numpy as np
from Loss_functions import Cosine_similarity, MSEP, AC_loss, MSE,compute_H_matrix
import matplotlib.pyplot as plt


p=1
neurons = [128, 256, 512]

layers = [1,2,3]

num_folds=5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
data_test, data_train, data_val = load_test_train_data()


input_size = len(data_train[0][0])
output_size = 50
dark_zone_mics_index=[0,1,2,3,4,5,6,7,8,9,10,11]
bright_zone_mics_index=[12]

            
if p==1:
    train_err_list=[]
    test_err_list=[]
    for neuron in neurons:
        fold_train_err = []
        fold_test_err= []
        print(f"\n--- Testing architecture: Layer 1: {neuron}")
        for folds in range(num_folds):
            data_test, data_train, data_val=load_test_train_data(random_seed=folds)
            data_test=data_test
            data_train=data_train

            X_train= data_train[0][0:50]
            X_test=data_test[0][0:50]
            X_train = X_train.to(torch.float32)
            X_test  = X_test.to(torch.float32)
            
            RIRs_train=data_train[5]
            RIRs_test=data_test[5]

            filters_train= data_train[1][0:50]
            filters_test=data_test[1][0:50]
            unique_filters_train, filter_indices_train = np.unique(filters_train, axis=0, return_inverse=True)
            filters_train = torch.from_numpy(filter_indices_train).long()  
            unique_filters_test, filter_indices_test = np.unique(filters_test, axis=0, return_inverse=True)
            filters_test = torch.from_numpy(filter_indices_test).long()  
            #model=model_.to(device)
            model = torch.nn.Sequential(
                torch.nn.Linear(input_size, neuron),
                torch.nn.ReLU(),
                torch.nn.Linear(neuron, len(unique_filters_train))
            )
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr = 1e-3)
            for epoch in range(1, 41):
                optimizer.zero_grad()
                outputs = model(X_train)
                loss = criterion(outputs, filters_train)
                loss.backward()
                optimizer.step()

            print("\n Training complete!")
            model.eval()
            with torch.no_grad():
                n_srcs = 3
                filter_len = 1024
                train_pred_logits = model(X_train)
                test_pred_logits = model(X_test)
                
                pred_classes_train = train_pred_logits.argmax(dim=1)
                pred_classes_test = test_pred_logits.argmax(dim=1)

                unique_filters_train = np.asarray(unique_filters_train)  # confirm numpy array
                unique_filters_test  = np.asarray(unique_filters_test)

                true_filters_train = torch.tensor(unique_filters_train[filters_train.numpy()], dtype=torch.float32)  # (batch, n_srcs*filter_len)
                true_filters_test  = torch.tensor(unique_filters_test[filters_test.numpy()], dtype=torch.float32)

                pred_filters_train = torch.tensor(unique_filters_train[pred_classes_train.numpy()], dtype=torch.float32)  # (batch, n_srcs*filter_len)
                pred_filters_test  = torch.tensor(unique_filters_test[pred_classes_test.numpy()], dtype=torch.float32)
                
                true_filters_train = true_filters_train.reshape(-1, n_srcs, filter_len)   # -> (batch, n_srcs, filter_len)
                true_filters_test  = true_filters_test.reshape(-1, n_srcs, filter_len)

                pred_filters_train = pred_filters_train.reshape(-1, n_srcs, filter_len)
                pred_filters_test  = pred_filters_test.reshape(-1, n_srcs, filter_len)

                filters_loss_train = []
                filters_loss_test  = []

                for i in range(len(pred_filters_train)):
                    feat_size = n_srcs * filter_len

                    pred_sample = pred_filters_train[i]   # (n_srcs, filter_len)
                    true_sample = true_filters_train[i]   # (n_srcs, filter_len)

                    pred_flat = pred_sample.reshape(1, -1)   # (1, feat_size)
                    true_flat = true_sample.reshape(1, -1)   # (1, feat_size)

                    L1_train = MSE(true_flat, pred_flat)   # scalar tensor
                    pred_sample_test = pred_filters_test[i]
                    true_sample_test = true_filters_test[i]
                    pred_flat_test = pred_sample_test.reshape(1, -1)
                    true_flat_test = true_sample_test.reshape(1, -1)
                    L1_test = MSE(true_flat_test, pred_flat_test)

                    L2_train = Cosine_similarity(pred_flat, true_flat)   # returns scalar
                    L2_test  = Cosine_similarity(pred_flat_test, true_flat_test)

                    rirs_train_i = RIRs_train[i]  # must be torch tensor (n_mics, n_srcs, n_rir_samples)
                    rirs_test_i  = RIRs_test[i]

                    if not isinstance(rirs_train_i, torch.Tensor):
                        rirs_train_i = torch.tensor(rirs_train_i, dtype=torch.float32)
                    if not isinstance(rirs_test_i, torch.Tensor):
                        rirs_test_i = torch.tensor(rirs_test_i, dtype=torch.float32)

                    x_input = torch.zeros(1, 1, dtype=torch.float32)   # adjust as required
                    x_input[0, 0] = 1.0

                    msep_B_train, msep_D_train = MSEP(pred_sample, true_sample, rirs_train_i, x_input, bright_zone_mics_index, dark_zone_mics_index)
                    msep_B_test,  msep_D_test  = MSEP(pred_sample_test, true_sample_test, rirs_test_i, x_input, bright_zone_mics_index, dark_zone_mics_index)

                    # If you only want bright-zone MSEP:
                    L3_train = msep_B_train
                    L3_test  = msep_B_test

                    L4_train = AC_loss(pred_sample, true_sample, compute_H_matrix(rirs_train_i)[0], bright_zone_mics_index, dark_zone_mics_index)
                    L4_test  = AC_loss(pred_sample_test, true_sample_test, compute_H_matrix(rirs_test_i)[0], bright_zone_mics_index, dark_zone_mics_index)

                    filters_loss_train.append( (L1_train + L2_train + L3_train + L4_train))
                    filters_loss_test.append( (L1_test  + L2_test  + L3_test  + L4_test))

                # Average across samples
                train_err = torch.stack(filters_loss_train).mean().item()
                test_err  = torch.stack(filters_loss_test).mean().item()

                fold_train_err.append(train_err)
                fold_test_err.append(test_err)
            print(f"Neuron: {neuron}, Fold {folds+1} -- Train error: {train_err:.4f}, Test error: {test_err:.4f}")
        print(f"Neuron: {neuron} -- Average Train error: {np.mean(fold_train_err):.4f}, "
          f"Average Test error: {np.mean(fold_test_err):.4f}\n")
        train_err_list.append(np.mean(fold_train_err))
        test_err_list.append(np.mean(fold_test_err))
    for neuronss,train,test in zip(neurons,train_err_list,test_err_list):
        print(f"Average over 5 folds: (Neurons, Train error, Test error): ({neuronss}, {train:.20f}, {test:.20f})")
        
if p == 2:
    neurons1 = [128, 256, 512]
    neurons2 = [128, 256, 512]

    test_err_grid = np.zeros((len(neurons1), len(neurons2)))
    train_err_grid = np.zeros((len(neurons1), len(neurons2)))

    for i, neuron1 in enumerate(neurons1):
        for j, neuron2 in enumerate(neurons2):

            print(f"\n--- Testing architecture: Layer 1: {neuron1}, Layer 2: {neuron2} ---")


            train_mse_folds = []
            test_mse_folds = []

            for fold in range(num_folds):

                print(f" Fold {fold+1}/{num_folds}")

                # Reload data each fold
                data_test, data_train, data_val = load_test_train_data(random_seed=fold)

                X_train = data_train[0][0:50].float()
                X_test  = data_test[0][0:50].float()

                RIRs_train = data_train[5]
                RIRs_test  = data_test[5]

                filters_train = data_train[1][0:50]
                filters_test  = data_test[1][0:50]

                unique_filters_train, filter_indices_train = np.unique(filters_train, axis=0, return_inverse=True)
                unique_filters_test,  filter_indices_test  = np.unique(filters_test, axis=0, return_inverse=True)

                filters_train = torch.from_numpy(filter_indices_train).long()
                filters_test  = torch.from_numpy(filter_indices_test).long()

                # --------------------
                # Build 2-hidden-layer model
                # --------------------
                model = torch.nn.Sequential(
                    torch.nn.Linear(input_size, neuron1),
                    torch.nn.ReLU(),
                    torch.nn.Linear(neuron1, neuron2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(neuron2, len(unique_filters_train))
                )
                
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr = 1e-3)
                # --------------------
                # Train model
                # --------------------
                for epoch in range(1, 41):
                    optimizer.zero_grad()
                    outputs = model(X_train)
                    loss = criterion(outputs, filters_train)
                    loss.backward()
                    optimizer.step()

                # --------------------
                # EVALUATION inside folds-loop  (your bug fix!)
                # --------------------
                model.eval()
                with torch.no_grad():

                    # Predict class indices
                    train_logits = model(X_train)
                    test_logits  = model(X_test)

                    pred_classes_train = train_logits.argmax(dim=1)
                    pred_classes_test  = test_logits.argmax(dim=1)

                    # Convert numpy filters → tensors
                    unique_filters_train = np.asarray(unique_filters_train)
                    unique_filters_test  = np.asarray(unique_filters_test)

                    # True filters (indexed)
                    true_filters_train = torch.tensor(unique_filters_train[filters_train.numpy()], dtype=torch.float32)
                    true_filters_test  = torch.tensor(unique_filters_test[filters_test.numpy()], dtype=torch.float32)

                    # Pred filters (indexed)
                    pred_filters_train = torch.tensor(unique_filters_train[pred_classes_train.numpy()], dtype=torch.float32)
                    pred_filters_test  = torch.tensor(unique_filters_test[pred_classes_test.numpy()], dtype=torch.float32)

                    # Reshape to (batch, 3, 1024)
                    n_srcs = 3
                    filter_len = 1024
                    true_filters_train = true_filters_train.reshape(-1, n_srcs, filter_len)
                    true_filters_test  = true_filters_test.reshape(-1, n_srcs, filter_len)
                    pred_filters_train = pred_filters_train.reshape(-1, n_srcs, filter_len)
                    pred_filters_test  = pred_filters_test.reshape(-1, n_srcs, filter_len)

                    # Compute error per sample
                    filters_loss_train = []
                    filters_loss_test  = []

                    for ii in range(len(pred_filters_train)):

                        pred_sample = pred_filters_train[ii]
                        true_sample = true_filters_train[ii]

                        pred_flat = pred_sample.reshape(1, -1)
                        true_flat = true_sample.reshape(1, -1)

                        # Base losses
                        L1_train = MSE(true_flat, pred_flat)
                        L2_train = Cosine_similarity(pred_flat, true_flat)

                        pred_sample_test = pred_filters_test[ii]
                        true_sample_test = true_filters_test[ii]

                        pred_flat_test = pred_sample_test.reshape(1, -1)
                        true_flat_test = true_sample_test.reshape(1, -1)

                        L1_test = MSE(true_flat_test, pred_flat_test)
                        L2_test = Cosine_similarity(pred_flat_test, true_flat_test)

                        # Pressure losses
                        rirs_train_i = torch.tensor(RIRs_train[ii], dtype=torch.float32)
                        rirs_test_i  = torch.tensor(RIRs_test[ii], dtype=torch.float32)

                        x_input = torch.zeros(1,1)
                        x_input[0,0] = 1.0

                        msep_B_train, _ = MSEP(pred_sample, true_sample, rirs_train_i, x_input, bright_zone_mics_index, dark_zone_mics_index)
                        msep_B_test,  _ = MSEP(pred_sample_test, true_sample_test, rirs_test_i, x_input, bright_zone_mics_index, dark_zone_mics_index)

                        L3_train = msep_B_train
                        L3_test  = msep_B_test

                        # AC loss
                        H_train = compute_H_matrix(rirs_train_i)[0]
                        H_test  = compute_H_matrix(rirs_test_i)[0]

                        L4_train = AC_loss(pred_sample, true_sample, H_train, bright_zone_mics_index, dark_zone_mics_index)
                        L4_test  = AC_loss(pred_sample_test, true_sample_test, H_test, bright_zone_mics_index, dark_zone_mics_index)

                        filters_loss_train.append((L1_train+L2_train+L3_train+L4_train))
                        filters_loss_test.append((L1_test+L2_test+L3_test+L4_test))

                    # Fold errors
                    train_mse_folds.append(torch.stack(filters_loss_train).mean().item())
                    test_mse_folds.append(torch.stack(filters_loss_test).mean().item())

            # ----------- END OF FOLDS LOOP -----------

            # Store average fold errors into heatmap grids
            train_err_grid[i, j] = np.mean(train_mse_folds)
            test_err_grid[i, j]  = np.mean(test_mse_folds)

    # -------------------------------------------------
    # PLOT HEATMAPS
    # -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def plot_mse_grid(ax, mse_grid, title):
        im = ax.imshow(mse_grid, origin='lower', cmap='viridis')
        for x in range(mse_grid.shape[0]):
            for y in range(mse_grid.shape[1]):
                ax.text(y, x, f"{mse_grid[x, y]:.2f}", ha='center', va='center', color='w',fontsize=15)
        ax.set_xticks(np.arange(len(neurons2)))
        ax.set_xticklabels(neurons2)
        ax.set_yticks(np.arange(len(neurons1)))
        ax.set_yticklabels(neurons1)
        ax.set_xlabel("Neurons in Layer 1", fontsize=12)
        ax.set_ylabel("Neurons in Layer 2", fontsize=12)
        ax.tick_params(axis='x',labelsize=12)
        ax.tick_params(axis='y',labelsize=12)
        ax.set_title(title)
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(title, fontsize=12)      # change label font size
        cbar.ax.tick_params(labelsize=12) 

    plot_mse_grid(axes[0], train_err_grid, "Train error")
    plot_mse_grid(axes[1], test_err_grid, "Test error")
    plt.tight_layout()
    plt.savefig(f"Plots/CV_classification_2_layers.pdf")
    plt.show()

if p == 3:
    neurons1 = [128, 256, 512]
    neurons2 = [128, 256, 512]
    neurons3 = [128, 256, 512]

    num_samples = len(data_train[0])
    test_err_grid = np.zeros((len(neurons1), len(neurons2), len(neurons3)))
    train_err_grid = np.zeros((len(neurons1), len(neurons2), len(neurons3)))

    for k, neuron3 in enumerate(neurons3):
        for i, neuron2 in enumerate(neurons2):
            for j, neuron1 in enumerate(neurons1):
                print(f"\n--- Testing architecture: Layer 1: {neuron1}, Layer 2: {neuron2}, Layer 3: {neuron3} ---")

                fold_train_err = []
                fold_test_err  = []

                for fold in range(num_folds):
                    data_test, data_train, data_val=load_test_train_data(random_seed=fold)
                    data_test=data_test
                    data_train=data_train

                    X_train= data_train[0][0:50]
                    X_test=data_test[0][0:50]
                    X_train = X_train.to(torch.float32)
                    X_test  = X_test.to(torch.float32)
                    
                    RIRs_train=data_train[5]
                    RIRs_test=data_test[5]
                    
                    filters_train= data_train[1][0:50]
                    filters_test=data_test[1][0:50]
                    unique_filters_train, filter_indices_train = np.unique(filters_train, axis=0, return_inverse=True)
                    filters_train = torch.from_numpy(filter_indices_train).long()  
                    unique_filters_test, filter_indices_test = np.unique(filters_test, axis=0, return_inverse=True)
                    filters_test = torch.from_numpy(filter_indices_test).long()  
                    # --- Build model ---
                    model = torch.nn.Sequential(
                        torch.nn.Linear(input_size, neuron1),
                        torch.nn.ReLU(),
                        torch.nn.Linear(neuron1, neuron2),
                        torch.nn.ReLU(),
                        torch.nn.Linear(neuron2, neuron3),
                        torch.nn.ReLU(),
                        torch.nn.Linear(neuron3, len(unique_filters_train))
                    )

                    criterion = nn.CrossEntropyLoss()
                    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
                    # --------------------
                    # Train model
                    # --------------------
                    for epoch in range(1, 41):
                        optimizer.zero_grad()
                        outputs = model(X_train)
                        loss = criterion(outputs, filters_train)
                        loss.backward()
                        optimizer.step()

                    # --- Evaluation ---
                    model.eval()
                    with torch.no_grad():
                        train_pred_logits = model(X_train)
                        test_pred_logits  = model(X_test)

                        pred_classes_train = train_pred_logits.argmax(dim=1)
                        pred_classes_test  = test_pred_logits.argmax(dim=1)

                        # Map predicted class indices to actual filters
                        unique_filters_np = np.asarray(unique_filters_train)
                        true_filters_train = torch.tensor(unique_filters_np[filters_train.numpy()], dtype=torch.float32)
                        true_filters_test  = torch.tensor(unique_filters_np[filters_test.numpy()], dtype=torch.float32)
                        pred_filters_train = torch.tensor(unique_filters_np[pred_classes_train.numpy()], dtype=torch.float32)
                        pred_filters_test  = torch.tensor(unique_filters_np[pred_classes_test.numpy()], dtype=torch.float32)

                        # Reshape filters: (batch, n_srcs, filter_len)
                        n_srcs = 3
                        filter_len = 1024
                        true_filters_train = true_filters_train.reshape(-1, n_srcs, filter_len)
                        true_filters_test  = true_filters_test.reshape(-1, n_srcs, filter_len)
                        pred_filters_train = pred_filters_train.reshape(-1, n_srcs, filter_len)
                        pred_filters_test  = pred_filters_test.reshape(-1, n_srcs, filter_len)

                        # Compute losses per sample
                        filters_loss_train = []
                        filters_loss_test  = []

                        for idx in range(len(pred_filters_train)):
                            # Flattened filters for MSE/Cosine
                            pred_flat  = pred_filters_train[idx].reshape(1, -1)
                            true_flat  = true_filters_train[idx].reshape(1, -1)
                            pred_flat_test = pred_filters_test[idx].reshape(1, -1)
                            true_flat_test = true_filters_test[idx].reshape(1, -1)

                            # L1: MSE
                            L1_train = MSE(true_flat, pred_flat)
                            L1_test  = MSE(true_flat_test, pred_flat_test)

                            # L2: Cosine similarity
                            L2_train = Cosine_similarity(pred_flat, true_flat)
                            L2_test  = Cosine_similarity(pred_flat_test, true_flat_test)

                            # L3: MSEP (Bright-zone only)
                            x_input = torch.zeros(1, 1)
                            x_input[0,0] = 1.0
                            rirs_train_i = torch.tensor(RIRs_train[idx], dtype=torch.float32)
                            rirs_test_i  = torch.tensor(RIRs_test[idx], dtype=torch.float32)
                            L3_train, _ = MSEP(pred_filters_train[idx], true_filters_train[idx],
                                               rirs_train_i, x_input, bright_zone_mics_index, dark_zone_mics_index)
                            L3_test,  _ = MSEP(pred_filters_test[idx], true_filters_test[idx],
                                               rirs_test_i, x_input, bright_zone_mics_index, dark_zone_mics_index)

                            # L4: AC_loss
                            L4_train = AC_loss(pred_filters_train[idx], true_filters_train[idx],
                                               compute_H_matrix(rirs_train_i)[0],
                                               bright_zone_mics_index, dark_zone_mics_index)
                            L4_test  = AC_loss(pred_filters_test[idx], true_filters_test[idx],
                                               compute_H_matrix(rirs_test_i)[0],
                                               bright_zone_mics_index, dark_zone_mics_index)

                            filters_loss_train.append((L1_train + L2_train + L3_train + L4_train))
                            filters_loss_test.append((L1_test + L2_test + L3_test + L4_test))

                        # Average across samples
                        fold_train_err.append(torch.stack([t.flatten() for t in filters_loss_train]).mean().item())
                        fold_test_err.append(torch.stack([t.flatten() for t in filters_loss_test]).mean().item())

                # Average over folds
                train_err_grid[j, i, k] = np.mean(fold_train_err)
                test_err_grid[j, i, k]  = np.mean(fold_test_err)

    with open("matrix_classification_train.txt", "w") as f:
        for slice2d in train_err_grid:
            np.savetxt(f, slice2d)
    train_err_grid = np.loadtxt("matrix_classification_train.txt").reshape(3,3,3)

    with open("matrix_classification_test.txt", "w") as f:
        for slice2d in test_err_grid:
            np.savetxt(f, slice2d)
    test_err_grid = np.loadtxt("matrix_classification_test.txt").reshape(3,3,3)
    

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2 rows: train/test, 3 cols: L3 neurons

    # Determine vmin/vmax separately for train and test
    vmin_train = train_err_grid.min()
    vmax_train = train_err_grid.max()
    vmin_test  = test_err_grid.min()
    vmax_test  = test_err_grid.max()

    for k, neuron3 in enumerate([128, 256, 512]):
        # Select slice for current L3 neuron
        train_slice = train_err_grid[:, :, k]  # shape (len(neurons1), len(neurons2))
        test_slice  = test_err_grid[:, :, k]

        # --- Plot Train Loss ---
        ax_train = axes[0, k]
        im_train = ax_train.imshow(train_slice, origin='lower', cmap='viridis', vmin=vmin_train, vmax=vmax_train)
        for x in range(len(neurons2)):
            for y in range(len(neurons1)):
                ax_train.text(y, x, f"{train_slice[x, y]:.2f}", ha='center', va='center', color='w', fontsize=15)
        ax_train.set_xticks(range(len(neurons1)))
        ax_train.set_xticklabels(neurons1)
        ax_train.set_yticks(range(len(neurons2)))
        ax_train.set_yticklabels(neurons2)
        ax_train.set_xlabel("Neurons in Layer 1", fontsize=12)
        ax_train.set_ylabel("Neurons in Layer 2", fontsize=12)
        ax_train.set_title(f"Train Loss, L3={neuron3}")
        ax_train.tick_params(axis='x',labelsize=12)
        ax_train.tick_params(axis='y',labelsize=12)

        # --- Plot Test Loss ---
        ax_test = axes[1, k]
        im_test = ax_test.imshow(test_slice, origin='lower', cmap='viridis', vmin=vmin_test, vmax=vmax_test)
        for x in range(len(neurons2)):
            for y in range(len(neurons1)):
                ax_test.text(y, x, f"{test_slice[x, y]:.2f}", ha='center', va='center', color='w', fontsize=15)
        ax_test.set_xticks(range(len(neurons1)))
        ax_test.set_xticklabels(neurons1)
        ax_test.set_yticks(range(len(neurons2)))
        ax_test.set_yticklabels(neurons2)
        ax_test.set_xlabel("Neurons in Layer 1", fontsize=12)
        ax_test.set_ylabel("Neurons in Layer 2", fontsize=12)
        ax_test.set_title(f"Test Loss, L3={neuron3}")
        ax_test.tick_params(axis='x',labelsize=12)
        ax_test.tick_params(axis='y',labelsize=12)

    # --- Add shared colorbars ---
    # Train colorbar
    # Train colorbar
    cbar_ax_train = fig.add_axes([0.92, 0.55, 0.02, 0.35])
    cbar_train = fig.colorbar(im_train, cax=cbar_ax_train)
    cbar_train.set_label('Train Loss', fontsize=12)   # <-- change font size here

    # Test colorbar
    cbar_ax_test = fig.add_axes([0.92, 0.1, 0.02, 0.35])
    cbar_test = fig.colorbar(im_test, cax=cbar_ax_test)
    cbar_test.set_label('Test Loss', fontsize=12)     # <-- and here


    plt.tight_layout(rect=[0,0,0.9,1])  # leave space for colorbars
    plt.savefig(f"Plots/CV_classification_3_layers.pdf")
    #plt.show()

