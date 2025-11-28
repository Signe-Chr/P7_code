import os, sys, torch, time
import numpy as np
import Cross_validation_models as cvm
from Dataset_class import CustomDataset
from torch.utils.data import DataLoader
from Dataset_generator_script import room_indices as ri
from tqdm import tqdm
from Train_test_split import load_test_train_data

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

data_dir = "ACC_filter_archive"
save_dir = "Saved Filters"
os.makedirs(save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_names = ("regression", "classification", "interpolation")

#---Load data and split into test and traning data---

#data_test=CustomDataset(data_dir,test_points)
#data_test_loader=DataLoader(data_test,batch_size=len(data_test), shuffle=True)
#temp_var_test=[batch for batch in data_test_loader][0]
#X_test=temp_var_test[0]
#data_train = CustomDataset(data_dir, train_points)
#data_train_loader = DataLoader(data_train, batch_size = len(data_train), shuffle=True)
#temp_var_train = [batch for batch in data_train_loader][0]
#YY = temp_var_train[1]
X_test, X_train = load_test_train_data(test_size=0.25, random_seed=42)
Y_test = X_test[1] #I tvivl om dette er korrekt!!!!!!!!!!!!!

def load_model(a):
    model_name = model_names[a]  # vælg model her
    input_size = 9
    output_file = os.path.join(save_dir, f"{model_name}_filters.pt")
    if model_name == "regression":
        output_size = 2160-540 #???????????
        model = cvm.FilterNet_regression(input_size, output_size).to(device)
        model.load_state_dict(torch.load("MLP_regression.pth", map_location=device))
    elif model_name == "classification":
        output_size = 324  # antal klasser ????????????????
        model = cvm.FilterNet_classification(input_size, output_size).to(device)
        model.load_state_dict(torch.load("MLP_classification.pth", map_location=device))
    elif model_name == "interpolation":
        output_size = 324
        model = cvm.FilterNet_interpolation(input_size, output_size).to(device)
        model.load_state_dict(torch.load("MLP_interpolation.pth", map_location=device))
    model.eval()
    return model, output_file

def generate_filters(a, X_test=X_test, Y_test=Y_test):
    model, output_file = load_model(a) # Choose model here (0-2)
    all_outputs = []

    for configuration in X_test:
        with torch.no_grad():
            output = model(configuration.unsqueeze(0).float())
            if a in [1, 2]:
                output = torch.matmul(Y_test.T.float(), output.T.float()).T

        # Gem output i dict
        all_outputs.append(output.cpu())

    # Gem alle output i én .pt fil
    torch.save(all_outputs, output_file)
    print(f"All outputs saved in '{output_file}'")

def test_model_efficiency(a, X_tests, device='cpu'):
    """
    Evaluates computational efficiency of a model on:
    - Inference time (avg per sample)
    - Model size (parameters + file size)
    - Algorithmic order (time scaling with input size)
    """
    # --- Load model ---
    model, output_file = load_model(a)
    model.to(device)
    model.eval()

    # --- Model size (parameters + file size) ---
    n_params = sum(p.numel() for p in model.parameters())
    torch.save(model.state_dict(), "tmp_model.pt")
    file_size_mb = os.path.getsize("tmp_model.pt") / (1024**2)
    os.remove("tmp_model.pt")

    # --- Runtime and scaling test ---
    times = []
    print(f"\nTesting computational efficiency for {len(X_tests)} input sizes...\n")

    for X_test in X_tests:
        start = time.perf_counter()
        with torch.no_grad():
            for x in tqdm(X_test, desc=f"Size {tuple(X_test.shape)}"):
                x = x.to(device).unsqueeze(0).float()
                output = model(x)
                if Y_test is not None:
                    output = torch.matmul(Y_test.T.float().to(device), output.T.float()).T
        if device == 'cuda':
            torch.cuda.synchronize()  # ensure GPU timing is accurate
        end = time.perf_counter()

        elapsed = end - start
        avg_time = elapsed / len(X_test)
        times.append((len(X_test), avg_time))

    # --- Estimate algorithmic order (log-log fit) ---
    n_vals = np.array([n for n, _ in times], dtype=float)
    t_vals = np.array([t for _, t in times], dtype=float)
    coeffs = np.polyfit(np.log(n_vals), np.log(t_vals), deg=1)
    k = coeffs[0]  # slope corresponds to order

    # --- Print summary ---
    print("\n--- Model Efficiency Summary ---")
    print(f"Total parameters: {n_params:,}")
    print(f"Model size: {file_size_mb:.3f} MB")
    print(f"Estimated algorithmic order: O(n^{k:.2f})")

    for n, t in times:
        print(f"Input size {n:>5}:  avg inference time = {t*1000:.3f} ms/sample")

    return {
        "n_params": n_params,
        "model_size_MB": file_size_mb,
        "algorithmic_order": k,
        "scaling_data": times
    }

for a in range(3):
    print(f"\n=== Evaluating Model {a} ({model_names[a]}) ===")
    test_model_efficiency(a, [X_test[:size] for size in [200]], device=device)

#generate_filters(0)  # vælg model her (0-2)
#generate_filters(1)
#generate_filters(2)
