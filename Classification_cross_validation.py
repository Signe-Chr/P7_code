import torch
import torch.nn.functional as F
import numpy as np
from torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
num_layers=3
num_epochs=200
np.random.seed(69420)
data = np.load("VAST_filter_archive.npy", allow_pickle=True).item()
number_of_folds=5

X_list, filters_list = [], []

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

# ---- 2. Prepare arrays and tensors
X = np.stack(X_list).astype(np.float32)        # [N_total, num_features]
filters = np.stack(filters_list).astype(np.float32)  # [N_total, filter_length]

num_total, input_size = X.shape
filter_length = filters.shape[1]

configs_tensor = torch.from_numpy(X)        # [N_total, num_features]
filters_tensor = torch.from_numpy(filters)  # [N_total, filter_length]
# Encode unique filters as integers
unique_filters, y_indices = np.unique(filters, axis=0, return_inverse=True)
y_tensor = torch.from_numpy(y_indices).long()  # [N_total]
num_classes = len(unique_filters)
print(num_classes)

print(f"Configs shape: {configs_tensor.shape}")
print(f"Filters shape: {filters_tensor.shape}") 

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

    all_test_mse_grids = []
    all_train_mse_grids = []

    for k, neuron3 in enumerate(neurons3):
        print(f'Neurons in 3rd layer {neuron3}')
        test_mse_grid = np.zeros((len(neurons2), len(neurons1)))
        train_mse_grid = np.zeros((len(neurons2), len(neurons1)))

        for i, neuron2 in enumerate(neurons2):
            print(f'Neurons in 2nd layer {neuron2}')
            for j, neuron1 in enumerate(neurons1):
                print(f'Neurons in 1st layer {neuron1}')

                fold_train_mse = []
                fold_test_mse = []

                # Cross-validation
                for fold in range(number_of_folds):
                    X_train, X_test, y_train, y_test = train_test_split(
                        configs_tensor, y_tensor, test_size=0.25, shuffle=True, random_state=fold
                    )

                    # --- Define model ---
                    model = torch.nn.Sequential(
                        torch.nn.Linear(input_size, neuron1),
                        torch.nn.ReLU(),
                        torch.nn.Linear(neuron1, neuron2),
                        torch.nn.ReLU(),
                        torch.nn.Linear(neuron2, neuron3),
                        torch.nn.ReLU(),
                        torch.nn.Linear(neuron3, num_classes)   # classification output
                    )
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
                        train_logits = model(X_train)
                        test_logits = model(X_test)

                        pred_classes_train = train_logits.argmax(dim=1)
                        pred_classes_test = test_logits.argmax(dim=1)

                        # Optional: compute MSE between predicted filter vectors
                        pred_filters_train = unique_filters[pred_classes_train.numpy()]
                        pred_filters_test = unique_filters[pred_classes_test.numpy()]

                        true_filters_train = unique_filters[y_train.numpy()]
                        true_filters_test = unique_filters[y_test.numpy()]

                        train_mse = np.mean((pred_filters_train - true_filters_train) ** 2)
                        test_mse = np.mean((pred_filters_test - true_filters_test) ** 2)

                        fold_train_mse.append(train_mse)
                        fold_test_mse.append(test_mse)

                # Average over folds
                train_mse_grid[i, j] = np.mean(fold_train_mse)
                test_mse_grid[i, j] = np.mean(fold_test_mse)

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


    