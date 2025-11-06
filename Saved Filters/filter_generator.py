import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import torch
import numpy as np
from Cross_validation_models import FilterNet_, model_  # brug din eksisterende model instans

data_dir = "Signes_data"
output_file = "Saved Filters/regression_filters.pt"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model (fra checkpoint)
input_size = len(model_[0][0]) if hasattr(model_, "__getitem__") else 9  # fallback 9
output_size = len(model_[0][1]) if hasattr(model_, "__getitem__") else model_.net[-1].out_features

model = FilterNet_(input_size, output_size).to(device)
model.load_state_dict(torch.load("MLP_regression_checkpoint.pth", map_location=device))
model.eval()

all_outputs = {}

for filename in os.listdir(data_dir):
    if filename.endswith(".npy"):
        file_path = os.path.join(data_dir, filename)
        data_dict = np.load(file_path, allow_pickle=True).item()

        # Byg input X
        X_new = np.concatenate([
            [data_dict.get('RT60', 0)],
            [data_dict.get('Phone_tilt', 0)],
            [data_dict.get('User_orientation', 0)],
            np.array(data_dict.get('Spatial_position', [0,0,0])).ravel(),
            np.array(data_dict.get('room_dim', [0,0,0]))
        ])

        X_tensor = torch.tensor(X_new, dtype=torch.float32).unsqueeze(0).to(device)  # batch=1

        # Forward pass
        with torch.no_grad():
            output = model(X_tensor)

        # Gem output i dict
        all_outputs[filename] = output.cpu()

# Gem alle output i én .pt fil
torch.save(all_outputs, output_file)
print(f"All outputs saved in '{output_file}'")
