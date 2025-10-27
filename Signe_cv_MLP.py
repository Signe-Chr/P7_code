import torch
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
num_layers=3
np.random.seed(69420)
# ---- 1. Load data from VAST archive
data = np.load("VAST_filter_archive.npy", allow_pickle=True).item()

X_list, filters_list,rt_60_list = [], [],[]

for key, inner in data.items():
    # Robust handling of features
    rt60 = inner.get('RT60', 0.0)
    phone_tilt = inner.get('Phone_tilt', 0.0)
    user_orient = inner.get('User_orientation', 0.0)
    spatial = np.array(inner.get('Spatial_position', [0, 0, 0]), dtype=np.float32).ravel()

    # Input feature vector
    X = np.concatenate([[rt60], [phone_tilt], [user_orient], spatial])
    X_list.append(X)

    # q_matrix = target filter coefficients
    q = inner.get('q_matrix', np.zeros(3072, dtype=np.float32))
    filters_list.append(q.flatten())
    
    rt_60_list.append(rt60)

# ---- 2. Prepare arrays and tensors
X = np.stack(X_list).astype(np.float32)        # [N_total, num_features]
filters = np.stack(filters_list).astype(np.float32)  # [N_total, filter_length]
rt60_array=np.array(rt_60_list)

num_total, input_size = X.shape
filter_length = filters.shape[1]

configs_tensor = torch.from_numpy(X)        # [N_total, num_features]
filters_tensor = torch.from_numpy(filters)  # [N_total, filter_length]

print(f"Configs shape: {configs_tensor.shape}")
print(f"Filters shape: {filters_tensor.shape}")



if num_layers == 1:
    neurons = [128, 256, 512]
    train_mse = []
    test_mse = []

    unique_rooms = np.unique(rt60_array)
    print("All rooms (RT60 values):", unique_rooms)

    for neuron in neurons:
        train_mse_fold = []
        test_mse_fold = []

        for test_room in unique_rooms:
            # --- Split data for cross-validation ---
            train_mask = rt60_array != test_room
            test_mask = rt60_array == test_room

            X_train = configs_tensor[train_mask]
            X_test = configs_tensor[test_mask]
            y_train = filters_tensor[train_mask]
            y_test = filters_tensor[test_mask]

            # --- Define model after y_train exists ---
            model = torch.nn.Sequential(
                torch.nn.Linear(input_size, neuron),
                torch.nn.ReLU(),
                torch.nn.Linear(neuron, y_train.shape[0])
            )
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

            # Optional: print model summary
            # summary(model, input_size=(input_size,))

            # --- Training loop ---
            model.train()
            for epoch in range(200):
                logits = model(X_train)
                weights = F.softmax(logits, dim=1)
                predicted_filters = weights @ y_train

                loss = criterion(predicted_filters, y_train)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # --- Evaluation ---
            model.eval()
            with torch.no_grad():
                train_pred = F.softmax(model(X_train), dim=1) @ y_train
                test_pred = F.softmax(model(X_test), dim=1) @ y_train

                train_mse_fold.append(torch.mean((train_pred - y_train) ** 2).item())
                test_mse_fold.append(torch.mean((test_pred - y_test) ** 2).item())

        # Average MSE across folds
        train_mse.append(np.mean(train_mse_fold))
        test_mse.append(np.mean(test_mse_fold))

    print(f"Average Training MSE for {len(unique_rooms)}-fold CV with 1 layer: {train_mse}")
    print(f"Average Test MSE for {len(unique_rooms)}-fold CV with 1 layer: {test_mse}")

    # --- Plot ---
    plt.plot(neurons, train_mse, label='Average Training MSE', marker='o')
    plt.plot(neurons, test_mse, linestyle='--', label='Average Test MSE', marker='o')
    plt.xlabel("Neurons in hidden layer")
    plt.ylabel("MSE")
    plt.title("1-layer MLP Cross-Validation")
    plt.grid(True)
    plt.legend()
    plt.show()


if num_layers == 2:
    neurons1 = [128, 256, 512]
    neurons2 = [128, 256, 512]
    test_mse_grid = np.zeros((len(neurons1), len(neurons2)))
    train_mse_grid = np.zeros((len(neurons1), len(neurons2)))

    for i, neuron1 in enumerate(neurons1):
        for j, neuron2 in enumerate(neurons2):

            # Cross-validation over rooms
            train_mse_folds = []
            test_mse_folds = []

            for test_room in np.unique(rt60_array):
                train_mask = rt60_array != test_room
                test_mask = rt60_array == test_room

                X_train = configs_tensor[train_mask]
                X_test = configs_tensor[test_mask]
                y_train = filters_tensor[train_mask]
                y_test = filters_tensor[test_mask]

                # --- Define model after y_train exists ---
                model = torch.nn.Sequential(
                    torch.nn.Linear(input_size, neuron1),
                    torch.nn.ReLU(),
                    torch.nn.Linear(neuron1, neuron2),
                    torch.nn.ReLU(),
                    torch.nn.Linear(neuron2, y_train.shape[0])
                )
                criterion = torch.nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                # Optional: print summary
                # summary(model, input_size=(input_size,))

                # --- Training loop ---
                model.train()
                for epoch in range(200):
                    logits = model(X_train)
                    weights = F.softmax(logits, dim=1)
                    predicted_filters = weights @ y_train

                    loss = criterion(predicted_filters, y_train)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # --- Evaluation ---
                model.eval()
                with torch.no_grad():
                    train_pred = F.softmax(model(X_train), dim=1) @ y_train
                    test_pred = F.softmax(model(X_test), dim=1) @ y_train

                    train_mse_folds.append(torch.mean((train_pred - y_train) ** 2).item())
                    test_mse_folds.append(torch.mean((test_pred - y_test) ** 2).item())

            # Average over folds
            train_mse_grid[i, j] = np.mean(train_mse_folds)
            test_mse_grid[i, j] = np.mean(test_mse_folds)

    # --- Plot heatmaps ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def plot_mse_grid(ax, mse_grid, title):
        im = ax.imshow(mse_grid, origin='lower', cmap='viridis')
        for x in range(mse_grid.shape[0]):
            for y in range(mse_grid.shape[1]):
                ax.text(y, x, f"{mse_grid[x, y]:.4f}", ha='center', va='center', color='w')
        ax.set_xticks(np.arange(len(neurons1)))
        ax.set_xticklabels(neurons1)
        ax.set_yticks(np.arange(len(neurons2)))
        ax.set_yticklabels(neurons2)
        ax.set_xlabel("Neurons in 1st layer")
        ax.set_ylabel("Neurons in 2nd layer")
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

    all_test_mse_grids = []
    all_train_mse_grids = []

    # --- 1. Compute all MSE grids ---
    for k, neuron3 in enumerate(neurons3):
        print(f'Neurons in 3rd layer {neuron3}')
        test_mse_grid = np.zeros((len(neurons2), len(neurons1)))   # y-axis: neurons2, x-axis: neurons1
        train_mse_grid = np.zeros((len(neurons2), len(neurons1)))

        for i, neuron2 in enumerate(neurons2):
            print(f'Neurons in 2nd layer {neuron2}')
            for j, neuron1 in enumerate(neurons1):
                print(f'Neurons in 1st layer {neuron1}')

                train_mse_folds = []
                test_mse_folds = []

                # Cross-validation over rooms
                for test_room in np.unique(rt60_array):
                    train_mask = rt60_array != test_room
                    test_mask = rt60_array == test_room

                    X_train = configs_tensor[train_mask]
                    X_test = configs_tensor[test_mask]
                    y_train = filters_tensor[train_mask]
                    y_test = filters_tensor[test_mask]

                    # --- Define model after y_train exists ---
                    model = torch.nn.Sequential(
                        torch.nn.Linear(input_size, neuron1),
                        torch.nn.ReLU(),
                        torch.nn.Linear(neuron1, neuron2),
                        torch.nn.ReLU(),
                        torch.nn.Linear(neuron2, neuron3),
                        torch.nn.ReLU(),
                        torch.nn.Linear(neuron3, y_train.shape[0])
                    )
                    criterion = torch.nn.MSELoss()
                    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

                    # --- Training loop ---
                    model.train()
                    for epoch in range(200):
                        logits = model(X_train)
                        weights = F.softmax(logits, dim=1)
                        predicted_filters = weights @ y_train

                        loss = criterion(predicted_filters, y_train)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                    # --- Evaluation ---
                    model.eval()
                    with torch.no_grad():
                        train_pred = F.softmax(model(X_train), dim=1) @ y_train
                        test_pred = F.softmax(model(X_test), dim=1) @ y_train
                        train_mse_folds.append(torch.mean((train_pred - y_train) ** 2).item())
                        test_mse_folds.append(torch.mean((test_pred - y_test) ** 2).item())

                # Average over folds
                train_mse_grid[i, j] = np.mean(train_mse_folds)
                test_mse_grid[i, j] = np.mean(test_mse_folds)

        all_train_mse_grids.append(train_mse_grid)
        all_test_mse_grids.append(test_mse_grid)

    # --- 2. Plotting ---
    fig = plt.figure(figsize=(14, 6))
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(2, len(neurons3), figure=fig, wspace=0.4, hspace=0.4)

    # Separate min/max for train and test
    test_min = min([g.min() for g in all_test_mse_grids])
    test_max = max([g.max() for g in all_test_mse_grids])
    train_min = min([g.min() for g in all_train_mse_grids])
    train_max = max([g.max() for g in all_train_mse_grids])

    for k, neuron3 in enumerate(neurons3):
        # Test MSE (top row)
        ax = fig.add_subplot(gs[0, k])
        im_test = ax.imshow(all_test_mse_grids[k], origin='lower', cmap='viridis', vmin=test_min, vmax=test_max)
        for x in range(len(neurons2)):
            for y in range(len(neurons1)):
                ax.text(y, x, f"{all_test_mse_grids[k][x, y]:.4f}", ha='center', va='center', color='w', fontsize=8)
        ax.set_xticks(range(len(neurons1)))
        ax.set_xticklabels(neurons1)
        ax.set_yticks(range(len(neurons2)))
        ax.set_yticklabels(neurons2)
        ax.set_xlabel("L1 neurons")
        ax.set_ylabel("L2 neurons")
        ax.set_title(f"Test MSE, L3={neuron3}")

        # Train MSE (bottom row)
        ax = fig.add_subplot(gs[1, k])
        im_train = ax.imshow(all_train_mse_grids[k], origin='lower', cmap='viridis', vmin=train_min, vmax=train_max)
        for x in range(len(neurons2)):
            for y in range(len(neurons1)):
                ax.text(y, x, f"{all_train_mse_grids[k][x, y]:.4f}", ha='center', va='center', color='w', fontsize=8)
        ax.set_xticks(range(len(neurons1)))
        ax.set_xticklabels(neurons1)
        ax.set_yticks(range(len(neurons2)))
        ax.set_yticklabels(neurons2)
        ax.set_xlabel("L1 neurons")
        ax.set_ylabel("L2 neurons")
        ax.set_title(f"Train MSE, L3={neuron3}")

    # Colorbar for test MSE (top row)
    cbar_ax_test = fig.add_axes([0.92, 0.55, 0.02, 0.35])
    fig.colorbar(im_test, cax=cbar_ax_test, label='Test MSE')

    # Colorbar for train MSE (bottom row)
    cbar_ax_train = fig.add_axes([0.92, 0.1, 0.02, 0.35])
    fig.colorbar(im_train, cax=cbar_ax_train, label='Train MSE')

    plt.show()










