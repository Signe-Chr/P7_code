import os, sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import torch
import numpy as np
import Cross_validation_models as cvm
model_ = cvm.model_



data_dir = "Signes_data"
save_dir = "Saved Filters"
os.makedirs(save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#Choose model
model_names = ("regression", "classification", "interpolation", "baseline", "random_selection")


def load_model(a):
    model_name = model_names[a]  # vælg model her
    output_file = os.path.join(save_dir, f"{model_name}_filters.pt")
    if model_name == "regression":
        # Load model (fra checkpoint)
        input_size = len(model_[0][0]) if hasattr(model_, "__getitem__") else 9  # fallback 9
        output_size = len(model_[0][1]) if hasattr(model_, "__getitem__") else model_.net[-1].out_features
        model = cvm.FilterNet_(input_size, output_size).to(device)
        model.load_state_dict(torch.load("MLP_regression_checkpoint.pth", map_location=device))
    elif model_name == "classification":
        input_size = 9      # antal features
        output_size = 2160  # antal klasser
        model = cvm.Classification_softmax(input_size, output_size).to(device)
        model.load_state_dict(torch.load("softmax_classifier.pth", map_location=device))
    elif model_name == "interpolation":
        input_size = 9
        output_size = 3072
        model = cvm.FilterNet_(input_size, output_size).to(device)
        model.load_state_dict(torch.load("MLP_interpolation_checkpoint.pth", map_location=device))
    elif model_name == "baseline":
        model = cvm.FilterNet_(input_size, output_size).to(device)
        model.load_state_dict(torch.load("baseline_checkpoint.pth", map_location=device))
    elif model_name == "random_selection":
        model = cvm.FilterNet_(input_size, output_size).to(device)
        model.load_state_dict(torch.load("MLP_regression_checkpoint.pth", map_location=device))
    model.eval()
    return model, output_file

def generate_filters(a):
    model, output_file = load_model(a) # Choose model here (0-4)
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

generate_filters(1)  # vælg model her (0-4)