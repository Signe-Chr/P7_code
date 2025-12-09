import numpy as np
import matplotlib.pyplot as plt




train_err_grid = np.loadtxt("matrix_interpolation_train_2_layers.txt")
test_err_grid = np.loadtxt("matrix_interpolation_test_2_layers.txt")

neurons1 = [128, 256, 512]
neurons2 = [128, 256, 512]

    # -------------------------------------------------
    # PLOT HEATMAPS
    # -------------------------------------------------

vmin_train = train_err_grid.min()
vmax_train = train_err_grid.max()
vmin_test  = test_err_grid.min()
vmax_test  = test_err_grid.max()

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

def plot_mse_grid(ax, mse_grid, title):
    im = ax.imshow(mse_grid.T, origin='lower', cmap='viridis')
    for x in range(mse_grid.shape[0]):
        for y in range(mse_grid.shape[1]):
            ax.text(y, x, f"{mse_grid[y, x]:.2f}", ha='center', va='center', color='w', fontsize=15)
    ax.set_xticks(np.arange(len(neurons1)))
    ax.set_xticklabels(neurons2)
    ax.set_yticks(np.arange(len(neurons2)))
    ax.set_yticklabels(neurons1)
    ax.set_xlabel("Neurons in layer 1", fontsize=12)
    ax.set_ylabel("Neurons in layer 2", fontsize=12)
    ax.tick_params(axis='y',labelsize=12)
    ax.tick_params(axis='x',labelsize=12)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

plot_mse_grid(axes[0], train_err_grid, "Train error")
plot_mse_grid(axes[1], test_err_grid, "Test error")
plt.tight_layout()
plt.savefig("Plots/CV_interpolation_2_layer_plot.pdf", dpi=500)
plt.show()