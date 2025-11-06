import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import torch
import numpy as np
import torch.nn as nn
import Cross_validation_models as cvm # tilpas til filnavnet med din klasse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Stier
data_dir = "Signes_data"
model_path = "MLP_regression_checkpoint.pth"

model = cvm.model_.to(device)

# Load model
state_dict = torch.load(model_path, weights_only = True)  # din gemte model
model.load_state_dict(state_dict)
model.eval()

# Vælg hvilket felt fra dict der skal bruges som input til modellen
input_field = 'q_matrix'  # skift til 'IR' eller andet hvis nødvendigt

for filename in os.listdir(data_dir):
    if filename.endswith(".npy"):
        file_path = os.path.join(data_dir, filename)
        
        # Load dict fra .npy fil
        data_dict = np.load(file_path, allow_pickle=True).item()
        
        # Hent numerisk array
        input_array = data_dict[input_field]
        
        # Konverter til tensor
        input_tensor = torch.from_numpy(input_array).float()
        
        # Tilføj batch dimension hvis nødvendigt
        if input_tensor.ndim == 2:  # fx (3, 1024) -> (1, 3, 1024)
            input_tensor = input_tensor.unsqueeze(0)
        elif input_tensor.ndim == 3:  # fx (13,3,512) -> (1,13,3,512)
            input_tensor = input_tensor.unsqueeze(0)
        
        # Forward pass
        with torch.no_grad():
            output = model(input_tensor)
        
        # Gem output som .pt
        output_file = os.path.join(data_dir, filename.replace(".npy", "_output.pt"))
        torch.save(output, output_file)
        print(f"Saved output for {filename} to {output_file}")

print("All files processed.")