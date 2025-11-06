import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import torch
import numpy as np
from Cross_validation_models import FilterNet

# Sti til data og model
data_dir = "Signes_data"
model_path = "filter_mlp_model_full.pth"  # Skift til din modelfil

# Load model
model = torch.load(model_path, weights_only=False)
model.eval()  # Sæt modellen i evaluerings-mode

# Loop igennem alle .npy filer i data_dir
for filename in os.listdir(data_dir):
    if filename.endswith(".npy"):
        file_path = os.path.join(data_dir, filename)
        
        # Load data
        data = np.load(file_path)
        # Konverter til PyTorch tensor (hvis nødvendigt, tilføj .float() eller .unsqueeze())
        data_tensor = torch.from_numpy(data).float()
        
        # Hvis modellen forventer batch dimension, tilføj dim
        if data_tensor.ndim == len(model.input_shape) - 1:  # eksempel: model input er (batch, features)
            data_tensor = data_tensor.unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            output = model(data_tensor)
        
        # Gem output som .pt fil
        output_file = os.path.join(data_dir, filename.replace(".npy", "_output.pt"))
        torch.save(output, output_file)
        print(f"Saved output for {filename} to {output_file}")

print("All files processed.")
