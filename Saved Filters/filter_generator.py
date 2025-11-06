import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import torch
import numpy as np
from Cross_validation_models import FilterNet_, model_  # brug din eksisterende model instans

import os
data_dir = "Signes_data"  # eller hvad din mappe hed
files = [f for f in os.listdir(data_dir) if f.endswith(".npy")]
print("Antal .npy filer i data_dir:", len(files))

# --- Indlæs filen ---
pt_path = "Saved Filters/regression_filters.pt"
data = torch.load(pt_path)

print(f"\n📦 Fil: {pt_path}")
print(f"🔢 Antal entries (forudsigelser): {len(data)}")

# --- Eksempel: print de første 5 filnavne ---
print("\n🗂  Første 5 nøgler (filnavne):")
for i, key in enumerate(list(data.keys())[:5]):
    print(f"  {i+1}. {key}")

# --- Tjek dimensioner på et par filtre ---
print("\n📏 Dimensioner på de første filtre:")
for i, key in enumerate(list(data.keys())[:3]):
    filt = data[key]
    shape = tuple(filt.shape)
    dtype = filt.dtype
    print(f"  • {key}: shape={shape}, dtype={dtype}")

# --- Tjek om alle filtre har samme form ---
shapes = [tuple(v.shape) for v in data.values()]
unique_shapes = set(shapes)

print(f"\n🔍 Unikke filter-shapes fundet: {unique_shapes}")
if len(unique_shapes) == 1:
    print(f"✅ Alle filtre har samme form: {unique_shapes.pop()}")
else:
    print("⚠️ Forskellige filter-dimensioner fundet!")

# --- Beregn samlet antal parametre per filter ---
first_filter = list(data.values())[0]
num_params = np.prod(first_filter.shape)
print(f"\n🧮 Hvert filter indeholder {num_params:,} værdier")

# --- Eventuelt: estimer samlet størrelse i hukommelsen ---
total_size = sum(v.numel() for v in data.values()) * 4 / (1024**2)
print(f"💾 Estimeret samlet størrelse: {total_size:.2f} MB\n")



data_dir = "Signes_data"
save_dir = "Saved Filters"
os.makedirs(save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model (fra checkpoint)
input_size = len(model_[0][0]) if hasattr(model_, "__getitem__") else 9  # fallback 9
output_size = len(model_[0][1]) if hasattr(model_, "__getitem__") else model_.net[-1].out_features

#Choose model
model_names = ("regression", "classification", "interpolation", "baseline", "random_selection")

def load_model(a):
    model_name = model_names[a]  # vælg model her
    output_file = os.path.join(save_dir, f"{model_name}_filters.pt")
    if model_name == "regression":
        model = FilterNet_(input_size, output_size).to(device)
        model.load_state_dict(torch.load("MLP_regression_checkpoint.pth", map_location=device))
    if model_name == "classification":
        model = FilterNet_(input_size, output_size).to(device)
        model.load_state_dict(torch.load("MLP_classification_checkpoint.pth", map_location=device))
    if model_name == "interpolation":
        model = FilterNet_(input_size, output_size).to(device)
        model.load_state_dict(torch.load("MLP_interpolation_checkpoint.pth", map_location=device))
    if model_name == "baseline":
        model = FilterNet_(input_size, output_size).to(device)
        model.load_state_dict(torch.load("baseline_checkpoint.pth", map_location=device))
    if model_name == "random_selection":
        model = FilterNet_(input_size, output_size).to(device)
        model.load_state_dict(torch.load("MLP_regression_checkpoint.pth", map_location=device))
    model.eval()
    return model, output_file

def generate_filters():
    model, output_file = load_model()
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
