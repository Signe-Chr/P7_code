
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from Test_train_split import load_test_train_data
from Loss_functions import Cosine_similarity, MSEP, AC_loss, MSE, compute_H_matrix

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Parameters
p = 3
neurons = [128, 256, 512]
layers = [1, 2, 3]
num_folds = 5

# Load data
data_test, data_train, nej = load_test_train_data()
input_size = len(data_train[0][0])

dark_zone_mics_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
bright_zone_mics_index = [12]

# x_input initialization
x_input = torch.zeros(1, 1, dtype=torch.float32).to(device)
x_input[0, 0] = 1.0

if p == 1:
    train_err_list = []
    test_err_list = []

    for neuron in neurons:
        fold_train_err = []
        fold_test_err = []
        print(f"\n--- Testing architecture: Layer 1: {neuron}")

        for folds in range(num_folds):
            # Reload data for each fold
            data_test, data_train, nej = load_test_train_data(random_seed=folds)

            # Prepare training and test sets
            X_train = data_train[0][:50].to(device).to(torch.float32)
            X_test = data_test[0][:50].to(device).to(torch.float32)

            filters_train = data_train[1][:50].to(device).to(torch.float32)
            filters_test = data_test[1][:50].to(device).to(torch.float32)

            RIRs_train = data_train[5][:50]
            RIRs_test = data_test[5][:50]

            output_size = len(X_train)

            # Define model and move to device
            model = torch.nn.Sequential(
                torch.nn.Linear(input_size, neuron),
                torch.nn.Dropout(0.3),
                torch.nn.ReLU(),
                torch.nn.Linear(neuron, output_size)
            ).to(device)

            optimizer = optim.Adam(model.parameters(), lr=1e-3)

            # Training loop
            for epoch in range(1, 41):
                print("epoch", epoch)
                total_loss = 0
                total = 0

                for i in range(len(X_train)):
                    X = X_train[i].unsqueeze(0).to(device)
                    filter = filters_train[i].unsqueeze(0).to(device)
                    rir = RIRs_train[i]

                    if not isinstance(rir, torch.Tensor):
                        rir = torch.tensor(rir, dtype=torch.float32)
                    rir = rir.to(device)

                    optimizer.zero_grad()

                    coeff = model(X)  # Already on device
                    outputs = torch.matmul(coeff, filters_train).to(device)

                    # 2D versions
                    filter_2d = filter.reshape(3, 1024).to(device)
                    outputs_2d = outputs.reshape(3, 1024).to(device)

                    H = compute_H_matrix(rir)[0].to(device)

                    loss = (
                        MSE(outputs, filter) +
                        Cosine_similarity(outputs, filter) +
                        MSEP(outputs_2d, filter_2d, rir, x_input, bright_zone_mics_index, dark_zone_mics_index)[0] +
                        AC_loss(outputs_2d, filter_2d, H, bright_zone_mics_index, dark_zone_mics_index)
                    )

                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item() * X.size(0)
                    total += X.size(0)

            avg_loss = total_loss / total
            print(" Training complete!")

            # Evaluation
            model.eval()
            with torch.no_grad():
                filters_loss_train = []
                filters_loss_test = []

                for i in range(len(filters_train)):
                    filter_train = filters_train[i].unsqueeze(0)
                    filter_test = filters_test[i].unsqueeze(0)

                    pred_filter_train = torch.matmul(model(X_train[i]), filters_train).unsqueeze(0)
                    pred_filter_test = torch.matmul(model(X_test[i]), filters_train).unsqueeze(0)

                    filter_train_2d = filter_train.reshape(3, 1024)
                    filter_test_2d = filter_test.reshape(3, 1024)
                    pred_filter_train_2d = pred_filter_train.reshape(3, 1024)
                    pred_filter_test_2d = pred_filter_test.reshape(3, 1024)

                    L1_train = MSE(pred_filter_train, filter_train)
                    L1_test = MSE(pred_filter_test, filter_test)

                    L2_train = Cosine_similarity(pred_filter_train, filter_train)
                    L2_test = Cosine_similarity(pred_filter_test, filter_test)

                    rirs_train_i = RIRs_train[i]
                    rirs_test_i = RIRs_test[i]

                    if not isinstance(rirs_train_i, torch.Tensor):
                        rirs_train_i = torch.tensor(rirs_train_i, dtype=torch.float32)
                    if not isinstance(rirs_test_i, torch.Tensor):
                        rirs_test_i = torch.tensor(rirs_test_i, dtype=torch.float32)

                    rirs_train_i = rirs_train_i.to(device)
                    rirs_test_i = rirs_test_i.to(device)

                    msep_B_train, msep_D_train = MSEP(pred_filter_train_2d, filter_train_2d, rirs_train_i, x_input, bright_zone_mics_index, dark_zone_mics_index)
                    msep_B_test, msep_D_test = MSEP(pred_filter_test_2d, filter_test_2d, rirs_test_i, x_input, bright_zone_mics_index, dark_zone_mics_index)

                    L3_train = msep_B_train
                    L3_test = msep_B_test

                    H_B_train = compute_H_matrix(rirs_train_i)[0].to(device)
                    H_B_test = compute_H_matrix(rirs_test_i)[0].to(device)

                    L4_train = AC_loss(pred_filter_train_2d, filter_train_2d, H_B_train, bright_zone_mics_index, dark_zone_mics_index)
                    L4_test = AC_loss(pred_filter_test_2d, filter_test_2d, H_B_test, bright_zone_mics_index, dark_zone_mics_index)

                    filters_loss_train.append(0.25 * (L1_train + L2_train + L3_train + L4_train))
                    filters_loss_test.append(0.25 * (L1_test + L2_test + L3_test + L4_test))

            train_err = torch.stack(filters_loss_train).mean().item()
            test_err = torch.stack(filters_loss_test).mean().item()

            fold_train_err.append(train_err)
            fold_test_err.append(test_err)

            print(f"Neuron: {neuron}, Fold {folds+1} -- Train error: {train_err:.4f}, Test error: {test_err:.4f}\n")

        print(f"Neuron: {neuron} -- Average Train error: {np.mean(fold_train_err):.4f}, "
              f"Average Test error: {np.mean(fold_test_err):.4f}\n")

        train_err_list.append(np.mean(fold_train_err))
        test_err_list.append(np.mean(fold_test_err))

    for neuronss, train, test in zip(neurons, train_err_list, test_err_list):
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
                # Reload data for each fold
                data_test, data_train, nej = load_test_train_data()

                # Prepare training and test sets
                X_train = data_train[0][:50].to(device).to(torch.float32)
                X_test = data_test[0][:50].to(device).to(torch.float32)

                filters_train = data_train[1][:50].to(device).to(torch.float32)
                filters_test = data_test[1][:50].to(device).to(torch.float32)

                RIRs_train = data_train[5][:50]
                RIRs_test = data_test[5][:50]

                output_size = len(X_train)

                # --------------------
                # Build 2-hidden-layer model
                # --------------------
                model = torch.nn.Sequential(
                    torch.nn.Linear(input_size, neuron1),
                    torch.nn.Dropout(0.3),
                    torch.nn.ReLU(),
                    torch.nn.Linear(neuron1, neuron2),
                    torch.nn.Dropout(0.3),
                    torch.nn.ReLU(),
                    torch.nn.Linear(neuron2, output_size)
                ).to(device)
                
                optimizer = optim.Adam(model.parameters(), lr = 1e-3)
                # --------------------
                # Train model
                # --------------------
                for epoch in range(1, 41):
                    total_loss = 0
                    total = 0
                    for k in range(len(X_train)):
                        X = X_train[k].unsqueeze(0).to(device)
                        optimizer.zero_grad()
                        coeff = model(X_train[k]).unsqueeze(0)
                        outputs = torch.matmul(coeff, filters_train).to(device)
                        filter = filters_train[k].unsqueeze(0).to(device)
                        rir = RIRs_train[k]
                        if not isinstance(rir, torch.Tensor):
                            rir = torch.tensor(rir, dtype=torch.float32)
                        rir = rir.to(device)

                        # 2D versions
                        filter_2d = filter.reshape(3, 1024)
                        outputs_2d = outputs.reshape(3, 1024)

                        H = compute_H_matrix(rir)[0].to(device)

                        loss =(
                            MSE(outputs, filter) +
                            Cosine_similarity(outputs, filter) +
                            MSEP(outputs_2d, filter_2d, rir, x_input, bright_zone_mics_index, dark_zone_mics_index)[0] +
                            AC_loss(outputs_2d, filter_2d, H, bright_zone_mics_index, dark_zone_mics_index))

                        loss.backward()
                        optimizer.step()

                        total_loss += loss.item() * X.size(0)
                        total += X.size(0)
                avg_loss = total_loss / total
                print(" Training complete!\n")
                # --------------------
                # EVALUATION inside folds-loop  (your bug fix!)
                # --------------------
                model.eval()
                with torch.no_grad():
                    filters_loss_train = []
                    filters_loss_test = []

                    for k in range(len(filters_train)):
                        filter_train = filters_train[k].unsqueeze(0)
                        filter_test = filters_test[k].unsqueeze(0)

                        pred_filter_train = torch.matmul(model(X_train[k]), filters_train).unsqueeze(0)
                        pred_filter_test = torch.matmul(model(X_test[k]), filters_train).unsqueeze(0)

                        filter_train_2d = filter_train.reshape(3, 1024)
                        filter_test_2d = filter_test.reshape(3, 1024)
                        pred_filter_train_2d = pred_filter_train.reshape(3, 1024)
                        pred_filter_test_2d = pred_filter_test.reshape(3, 1024)

                        L1_train = MSE(pred_filter_train, filter_train)
                        L1_test = MSE(pred_filter_test, filter_test)

                        L2_train = Cosine_similarity(pred_filter_train, filter_train)
                        L2_test = Cosine_similarity(pred_filter_test, filter_test)

                        rirs_train_i = RIRs_train[k]
                        rirs_test_i = RIRs_test[k]

                        if not isinstance(rirs_train_i, torch.Tensor):
                            rirs_train_i = torch.tensor(rirs_train_i, dtype=torch.float32)
                        if not isinstance(rirs_test_i, torch.Tensor):
                            rirs_test_i = torch.tensor(rirs_test_i, dtype=torch.float32)

                        rirs_train_i = rirs_train_i.to(device)
                        rirs_test_i = rirs_test_i.to(device)

                        msep_B_train, msep_D_train = MSEP(pred_filter_train_2d, filter_train_2d, rirs_train_i, x_input, bright_zone_mics_index, dark_zone_mics_index)
                        msep_B_test, msep_D_test = MSEP(pred_filter_test_2d, filter_test_2d, rirs_test_i, x_input, bright_zone_mics_index, dark_zone_mics_index)

                        L3_train = msep_B_train
                        L3_test = msep_B_test

                        H_B_train = compute_H_matrix(rirs_train_i)[0].to(device)
                        H_B_test = compute_H_matrix(rirs_test_i)[0].to(device)

                        L4_train = AC_loss(pred_filter_train_2d, filter_train_2d, H_B_train, bright_zone_mics_index, dark_zone_mics_index)
                        L4_test = AC_loss(pred_filter_test_2d, filter_test_2d, H_B_test, bright_zone_mics_index, dark_zone_mics_index)

                        filters_loss_train.append(0.25 * (L1_train + L2_train + L3_train + L4_train))
                        filters_loss_test.append(0.25 * (L1_test + L2_test + L3_test + L4_test))

                    # Fold errors
                    train_mse_folds.append(torch.stack(filters_loss_train).mean().item())
                    test_mse_folds.append(torch.stack(filters_loss_test).mean().item())

            # ----------- END OF FOLDS LOOP -----------

            # Store average fold errors into heatmap grids
            train_err_grid[i, j] = np.mean(train_mse_folds)
            test_err_grid[i, j]  = np.mean(test_mse_folds)
    np.savetxt("matrix_interpolation_train_2_layers.txt", train_err_grid)

    np.savetxt("matrix_interpolation_test_2_layers.txt", test_err_grid)

    train_err_grid = np.loadtxt("matrix_interpolation_train_2_layers.txt").reshape(3,3)
    test_err_grid = np.loadtxt("matrix_interpolation_test_2_layers.txt").reshape(3,3)

    exit()

    # -------------------------------------------------
    # PLOT HEATMAPS
    # -------------------------------------------------
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

    plot_mse_grid(axes[0], train_err_grid, "Train error")
    plot_mse_grid(axes[1], test_err_grid, "Test error")
    plt.tight_layout()
    plt.show()

if p == 3:
    neurons1 = [128, 256, 512]
    neurons2 = [128, 256, 512]
    neurons3 = [128, 256, 512]

    test_err_grid = np.zeros((len(neurons1), len(neurons2), len(neurons3)))
    train_err_grid = np.zeros((len(neurons1), len(neurons2), len(neurons3)))

    """for u, neuron1 in enumerate(neurons1):
        for i, neuron2 in enumerate(neurons2):
            for j, neuron3 in enumerate(neurons3):
                print(f"\n--- Testing architecture: Layer 1: {neuron1}, Layer 2: {neuron2}, Layer 3 {neuron3} ---")

                train_mse_folds = []
                test_mse_folds = []

                for fold in range(num_folds):
                    # Reload data for each fold
                    data_test, data_train, nej = load_test_train_data()

                    # Prepare training and test sets
                    X_train = data_train[0][:50].to(device).to(torch.float32)
                    X_test = data_test[0][:50].to(device).to(torch.float32)

                    filters_train = data_train[1][:50].to(device).to(torch.float32)
                    filters_test = data_test[1][:50].to(device).to(torch.float32)

                    RIRs_train = data_train[5][:50]
                    RIRs_test = data_test[5][:50]

                    output_size = len(X_train)

                    # --------------------
                    # Build 2-hidden-layer model
                    # --------------------
                    model = torch.nn.Sequential(
                        torch.nn.Linear(input_size, neuron1),
                        torch.nn.Dropout(0.3),
                        torch.nn.ReLU(),
                        torch.nn.Linear(neuron1, neuron2),
                        torch.nn.Dropout(0.3),
                        torch.nn.ReLU(),
                        torch.nn.Linear(neuron2, neuron3),
                        torch.nn.Dropout(0.3),
                        torch.nn.ReLU(),
                        torch.nn.Linear(neuron3, output_size)
                    ).to(device)
                    
                    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
                    # --------------------
                    # Train model
                    # --------------------
                    for epoch in range(1, 41):
                        total_loss = 0
                        total = 0
                        for k in range(len(X_train)):
                            X = X_train[k].unsqueeze(0).to(device)
                            optimizer.zero_grad()
                            coeff = model(X_train[k]).unsqueeze(0).to(device)
                            outputs = torch.matmul(coeff, filters_train).to(device)
                            filter = filters_train[k].unsqueeze(0).to(device)
                            rir = RIRs_train[k]
                            if not isinstance(rir, torch.Tensor):
                                rir = torch.tensor(rir, dtype=torch.float32)
                            rir = rir.to(device)

                            # 2D versions
                            filter_2d = filter.reshape(3, 1024)
                            outputs_2d = outputs.reshape(3, 1024)

                            H = compute_H_matrix(rir)[0].to(device)

                            loss = (
                                MSE(outputs, filter) +
                                Cosine_similarity(outputs, filter) +
                                MSEP(outputs_2d, filter_2d, rir, x_input, bright_zone_mics_index, dark_zone_mics_index)[0] +
                                AC_loss(outputs_2d, filter_2d, H, bright_zone_mics_index, dark_zone_mics_index)
                            )

                            loss.backward()
                            optimizer.step()

                            total_loss += loss.item() * X.size(0)
                            total += X.size(0)
                    avg_loss = total_loss / total
                    print(" Training complete!\n")
                    # --------------------
                    # EVALUATION inside folds-loop  (your bug fix!)
                    # --------------------
                    model.eval()
                    with torch.no_grad():
                        filters_loss_train = []
                        filters_loss_test = []

                        for k in range(len(filters_train)):
                            filter_train = filters_train[k].unsqueeze(0)
                            filter_test = filters_test[k].unsqueeze(0)

                            pred_filter_train = torch.matmul(model(X_train[k]), filters_train).unsqueeze(0)
                            pred_filter_test = torch.matmul(model(X_test[k]), filters_test).unsqueeze(0)


                            filter_train_2d = filter_train.reshape(3, 1024)
                            filter_test_2d = filter_test.reshape(3, 1024)
                            pred_filter_train_2d = pred_filter_train.reshape(3, 1024)
                            pred_filter_test_2d = pred_filter_test.reshape(3, 1024)

                            L1_train = MSE(pred_filter_train, filter_train)
                            L1_test = MSE(pred_filter_test, filter_test)

                            L2_train = Cosine_similarity(pred_filter_train, filter_train)
                            L2_test = Cosine_similarity(pred_filter_test, filter_test)

                            rirs_train_i = RIRs_train[k]
                            rirs_test_i = RIRs_test[k]

                            if not isinstance(rirs_train_i, torch.Tensor):
                                rirs_train_i = torch.tensor(rirs_train_i, dtype=torch.float32)
                            if not isinstance(rirs_test_i, torch.Tensor):
                                rirs_test_i = torch.tensor(rirs_test_i, dtype=torch.float32)

                            rirs_train_i = rirs_train_i.to(device)
                            rirs_test_i = rirs_test_i.to(device)

                            msep_B_train, msep_D_train = MSEP(pred_filter_train_2d, filter_train_2d, rirs_train_i, x_input, bright_zone_mics_index, dark_zone_mics_index)
                            msep_B_test, msep_D_test = MSEP(pred_filter_test_2d, filter_test_2d, rirs_test_i, x_input, bright_zone_mics_index, dark_zone_mics_index)

                            L3_train = msep_B_train
                            L3_test = msep_B_test

                            H_B_train = compute_H_matrix(rirs_train_i)[0].to(device)
                            H_B_test = compute_H_matrix(rirs_test_i)[0].to(device)

                            L4_train = AC_loss(pred_filter_train_2d, filter_train_2d, H_B_train, bright_zone_mics_index, dark_zone_mics_index)
                            L4_test = AC_loss(pred_filter_test_2d, filter_test_2d, H_B_test, bright_zone_mics_index, dark_zone_mics_index)

                            filters_loss_train.append((L1_train + L2_train + L3_train + L4_train))
                            filters_loss_test.append((L1_test + L2_test + L3_test + L4_test))

                        # Fold errors
                        train_mse_folds.append(torch.stack(filters_loss_train).mean().item())
                        test_mse_folds.append(torch.stack(filters_loss_test).mean().item())

                # ----------- END OF FOLDS LOOP -----------

                # Store average fold errors into heatmap grids
                train_err_grid[u, i, j] = np.mean(train_mse_folds)
                test_err_grid[u, i, j]  = np.mean(test_mse_folds)
    
    with open("matrix_interpolation_train.txt", "w") as f:
        for slice2d in train_err_grid:
            np.savetxt(f, slice2d)

    with open("matrix_interpolation_test.txt", "w") as f:
        for slice2d in test_err_grid:
            np.savetxt(f, slice2d)"""

    train_err_grid = np.loadtxt("matrix_interpolation_train_3_layers.txt").reshape(3,3,3)
    test_err_grid = np.loadtxt("matrix_interpolation_test_3_layers.txt").reshape(3,3,3)

    # -------------------------------------------------
    # PLOT HEATMAPS
    # -------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(14, 6))

    # Determine vmin/vmax separately for train and test
    vmin_train = train_err_grid.min()
    vmax_train = train_err_grid.max()
    vmin_test  = test_err_grid.min()
    vmax_test  = test_err_grid.max()
    

    for u, neuron3 in enumerate([128, 256, 512]):
        # Select slice for current L3 neuron

        train_slice = train_err_grid[:, :, u]
        test_slice  = test_err_grid[:, :, u]

        # --- Plot Train Loss ---
        ax_train = axes[0, u]
        im_train = ax_train.imshow(train_slice.T, origin='lower', cmap='viridis', vmin=vmin_train, vmax=vmax_train)
        for x in range(len(neurons1)):
            for y in range(len(neurons2)):
                ax_train.text(y, x, f"{train_slice[y, x]:.4f}", ha='center', va='center', color='w', fontsize=8)
        ax_train.set_xticks(range(len(neurons2)))
        ax_train.set_xticklabels(neurons2)
        ax_train.set_yticks(range(len(neurons1)))
        ax_train.set_yticklabels(neurons1)
        ax_train.set_xlabel("Neurons in Layer 1")
        ax_train.set_ylabel("Neurons in Layer 2")
        ax_train.set_title(f"Train Loss, L3={neuron3}")

        # --- Plot Test Loss ---
        ax_test = axes[1, u]
        im_test = ax_test.imshow(test_slice.T, origin='lower', cmap='viridis', vmin=vmin_test, vmax=vmax_test)
        for x in range(len(neurons1)):
            for y in range(len(neurons2)):
                ax_test.text(y, x, f"{test_slice[y, x]:.4f}", ha='center', va='center', color='w', fontsize=8)
        ax_test.set_xticks(range(len(neurons2)))
        ax_test.set_xticklabels(neurons2)
        ax_test.set_yticks(range(len(neurons1)))
        ax_test.set_yticklabels(neurons1)
        ax_test.set_xlabel("Neurons in Layer 1")
        ax_test.set_ylabel("Neurons in Layer 2")
        ax_test.set_title(f"Test Loss, L3={neuron3}")

    # --- Add shared colorbars ---
    # Train colorbar
    cbar_ax_train = fig.add_axes([0.92, 0.55, 0.02, 0.35])  # [left, bottom, width, height]
    fig.colorbar(im_train, cax=cbar_ax_train, label='Train Loss')

    # Test colorbar
    cbar_ax_test = fig.add_axes([0.92, 0.1, 0.02, 0.35])
    fig.colorbar(im_test, cax=cbar_ax_test, label='Test Loss')

    plt.tight_layout(rect=[0,0,0.9,1])  # leave space for colorbars
    plt.savefig("Plots/CV_interpolation_3_layers.pdf", dpi = 500)
