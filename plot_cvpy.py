import os, sys, torch
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LinearSegmentedColormap

# Dine farver
colors = ["#5B3758",  # Primary Lilla
          "#00916E",  # Secondary Mørkegrøn
         # "#DE6C83",  # Accent Pink
          "#FCB97D",  # Contrast Orange
          "#D4E4BC"]  # Contrast Lysegrøn

# Lav en glidende colormap
cmap = LinearSegmentedColormap.from_list("mycmap", colors, N=256)

train_err_grid_2 = np.loadtxt("CV filer/cross_validation_classification_train_2_layers.txt")
test_err_grid_2 = np.loadtxt("CV filer/cross_validation_classification_test_2_layers.txt")
# -------------------------------------------------
# PLOT HEATMAPS
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
neurons1 = [128, 256, 512]
neurons2 = [128, 256, 512]
def plot_mse_grid(ax, mse_grid, title):
    im = ax.imshow(mse_grid.T, origin='lower', cmap=cmap)
    for x in range(mse_grid.shape[0]):
        for y in range(mse_grid.shape[1]):
            ax.text(y, x, f"{mse_grid[y, x]:.2f}", ha='center', va='center', color='w',fontsize=15)
    ax.set_xticks(np.arange(len(neurons2)))
    ax.set_xticklabels(neurons2)
    ax.set_yticks(np.arange(len(neurons1)))
    ax.set_yticklabels(neurons1)
    ax.set_xlabel("Neurons in Layer 1", fontsize=18)
    ax.set_ylabel("Neurons in Layer 2", fontsize=18)
    ax.tick_params(axis='x',labelsize=18)
    ax.tick_params(axis='y',labelsize=18)
    ax.set_title(title,fontsize=18)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(title, fontsize=18)      # change label font size
    cbar.ax.tick_params(labelsize=18) 

plot_mse_grid(axes[0], train_err_grid_2, "Train error")
plot_mse_grid(axes[1], test_err_grid_2, "Test error")
plt.tight_layout()
plt.savefig(f"Plots/CV_classification_2_layers.pdf")
plt.show()

train_err_grid_3 = np.loadtxt("CV filer/matrix_classification_train.txt").reshape(3,3,3)
test_err_grid_3 = np.loadtxt("CV filer/matrix_classification_test.txt").reshape(3,3,3)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))  # 2 rows: train/test, 3 cols: L3 neurons

# Determine vmin/vmax separately for train and test
vmin_train = train_err_grid_3.min()
vmax_train = train_err_grid_3.max()
vmin_test  = test_err_grid_3.min()
vmax_test  = test_err_grid_3.max()

for k, neuron3 in enumerate([128, 256, 512]):
    # Select slice for current L3 neuron
    train_slice = train_err_grid_3[:, :, k]  # shape (len(neurons1), len(neurons2))
    test_slice  = test_err_grid_3[:, :, k]

    # --- Plot Train Loss ---
    ax_train = axes[0, k]
    im_train = ax_train.imshow(train_slice.T, origin='lower', cmap=cmap, vmin=vmin_train, vmax=vmax_train)
    for x in range(len(neurons2)):
        for y in range(len(neurons1)):
            ax_train.text(y, x, f"{train_slice[y, x]:.2f}", ha='center', va='center', color='w', fontsize=18)
    ax_train.set_xticks(range(len(neurons1)))
    ax_train.set_xticklabels(neurons1)
    ax_train.set_yticks(range(len(neurons2)))
    ax_train.set_yticklabels(neurons2)
    ax_train.set_xlabel("Neurons in Layer 1", fontsize=18)
    ax_train.set_ylabel("Neurons in Layer 2", fontsize=18)
    ax_train.set_title(f"Train Loss, L3={neuron3}",fontsize=18)
    ax_train.tick_params(axis='x',labelsize=18)
    ax_train.tick_params(axis='y',labelsize=18)

    # --- Plot Test Loss ---
    ax_test = axes[1, k]
    im_test = ax_test.imshow(test_slice.T, origin='lower', cmap=cmap, vmin=vmin_test, vmax=vmax_test)
    for x in range(len(neurons2)):
        for y in range(len(neurons1)):
            ax_test.text(y, x, f"{test_slice[y, x]:.2f}", ha='center', va='center', color='w', fontsize=18)
    ax_test.set_xticks(range(len(neurons1)))
    ax_test.set_xticklabels(neurons1)
    ax_test.set_yticks(range(len(neurons2)))
    ax_test.set_yticklabels(neurons2)
    ax_test.set_xlabel("Neurons in Layer 1", fontsize=18)
    ax_test.set_ylabel("Neurons in Layer 2", fontsize=18)
    ax_test.set_title(f"Test Loss, L3={neuron3}",fontsize=18)
    ax_test.tick_params(axis='x',labelsize=18)
    ax_test.tick_params(axis='y',labelsize=18)

# --- Add shared colorbars ---
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
plt.show()

train_err_grid_4 = np.loadtxt("CV filer/matrix_interpolation_train_2_layers.txt").reshape(3,3)
test_err_grid_4 = np.loadtxt("CV filer/matrix_interpolation_test_2_layers.txt").reshape(3,3)



# -------------------------------------------------
# PLOT HEATMAPS
# -------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

def plot_mse_grid(ax, mse_grid, title):
    im = ax.imshow(mse_grid, origin='lower', cmap=cmap)
    for x in range(mse_grid.shape[0]):
        for y in range(mse_grid.shape[1]):
            ax.text(y, x, f"{mse_grid[x, y]:.2f}", ha='center', va='center', color='w',fontsize=15)
    ax.set_xticks(np.arange(len(neurons2)))
    ax.set_xticklabels(neurons2)
    ax.set_yticks(np.arange(len(neurons1)))
    ax.set_yticklabels(neurons1)
    ax.tick_params(axis='x',labelsize=18)
    ax.tick_params(axis='y',labelsize=18)
    ax.set_xlabel("Neurons in 2nd layer",fontsize=18)
    ax.set_ylabel("Neurons in 1st layer",fontsize=18)
    ax.set_title(title,fontsize=18)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(title, fontsize=18)      # change label font size
    cbar.ax.tick_params(labelsize=18)
plot_mse_grid(axes[0], train_err_grid_4, "Train error")
plot_mse_grid(axes[1], test_err_grid_4, "Test error")
plt.tight_layout()
plt.savefig(f"Plots/CV_interpolation_2_layers.pdf")
plt.show()

train_err_grid_5 = np.loadtxt("CV filer/matrix_interpolation_train_3_layers.txt").reshape(3,3,3)
test_err_grid_5 = np.loadtxt("CV filer/matrix_interpolation_test_3_layers.txt").reshape(3,3,3)

# -------------------------------------------------
# PLOT HEATMAPS
# -------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Determine vmin/vmax separately for train and test
vmin_train = train_err_grid_5.min()
vmax_train = train_err_grid_5.max()
vmin_test  = test_err_grid_5.min()
vmax_test  = test_err_grid_5.max()


for u, neuron3 in enumerate([128, 256, 512]):
    # Select slice for current L3 neuron

    train_slice = train_err_grid_5[:, :, u]
    test_slice  = test_err_grid_5[:, :, u]

    # --- Plot Train Loss ---
    ax_train = axes[0, u]
    im_train = ax_train.imshow(train_slice.T, origin='lower', cmap=cmap, vmin=vmin_train, vmax=vmax_train)
    for x in range(len(neurons1)):
        for y in range(len(neurons2)):
            ax_train.text(y, x, f"{train_slice[y, x]:.2f}", ha='center', va='center', color='w', fontsize=18)
    ax_train.set_xticks(range(len(neurons2)))
    ax_train.set_xticklabels(neurons2)
    ax_train.set_yticks(range(len(neurons1)))
    ax_train.set_yticklabels(neurons1)
    ax_train.set_xlabel("Neurons in Layer 1", fontsize=18)
    ax_train.set_ylabel("Neurons in Layer 2", fontsize=18)
    ax_train.set_title(f"Train Loss, L3={neuron3}",fontsize=18)
    ax_train.tick_params(axis='x',labelsize=18)
    ax_train.tick_params(axis='y',labelsize=18)

    # --- Plot Test Loss ---
    ax_test = axes[1, u]
    im_test = ax_test.imshow(test_slice.T, origin='lower', cmap=cmap, vmin=vmin_test, vmax=vmax_test)
    for x in range(len(neurons1)):
        for y in range(len(neurons2)):
            ax_test.text(y, x, f"{test_slice[y, x]:.2f}", ha='center', va='center', color='w', fontsize=18)
    ax_test.set_xticks(range(len(neurons2)))
    ax_test.set_xticklabels(neurons2)
    ax_test.set_yticks(range(len(neurons1)))
    ax_test.set_yticklabels(neurons1)
    ax_test.set_xlabel("Neurons in Layer 1", fontsize=18)
    ax_test.set_ylabel("Neurons in Layer 2", fontsize=18)
    ax_test.set_title(f"Test Loss, L3={neuron3}",fontsize=18)
    ax_test.tick_params(axis='x',labelsize=18)
    ax_test.tick_params(axis='y',labelsize=18)

# --- Add shared colorbars ---
# Train colorbar
cbar_ax_train = fig.add_axes([0.92, 0.55, 0.02, 0.35])  # [left, bottom, width, height]
cbar_train = fig.colorbar(im_train, cax=cbar_ax_train)
cbar_train.set_label('Train Loss', fontsize=12)     # <-- and here

# Test colorbar
cbar_ax_test = fig.add_axes([0.92, 0.1, 0.02, 0.35])
cbar_test = fig.colorbar(im_test, cax=cbar_ax_test)
cbar_test.set_label('Test Loss', fontsize=12)     # <-- and here

plt.tight_layout(rect=[0,0,0.9,1])  # leave space for colorbars
plt.savefig("Plots/CV_interpolation_3_layers.pdf", dpi = 500)
plt.show()