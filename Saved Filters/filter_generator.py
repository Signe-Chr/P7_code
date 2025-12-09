import os, sys, torch, time
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import numpy as np
import Cross_validation_models as cvm
from tqdm import tqdm
from Test_train_split import load_test_train_data





def load_model(model_name):
    input_size = 9
    output_file = os.path.join(save_dir, f"{model_name}_filters.pt")
    if model_name == "regression":
        output_size = filters_train.shape[1]
        model = cvm.FilterNet_regression(input_size, output_size).to(device)
        model.load_state_dict(torch.load("MLP_regression.pth", map_location=device))
    elif model_name == "classification":
        output_size = filters_train.shape[0] 
        model = cvm.FilterNet_classification(input_size, output_size).to(device)
        model.load_state_dict(torch.load("MLP_classification_500.pth", map_location=device))
    elif model_name == "interpolation":
        output_size = filters_train.shape[0]
        model = cvm.FilterNet_interpolation(input_size, output_size).to(device)
        model.load_state_dict(torch.load("MLP_interpolation_without_weights.pth", map_location=device))
    model.eval()
    return model, output_file

def generate_filters(chosen_model, X_test, Y_train):
    model, output_file = load_model(chosen_model)
    all_outputs = []
    timings = []
    for configuration in X_test:
        with torch.no_grad():
            start = time.perf_counter()
            output = model(configuration.unsqueeze(0).float())
            if chosen_model == "classification":
                output = _, prediction = torch.max(output, 1)
                output = Y_train[prediction]
            if chosen_model == "interpolation":
                output = torch.matmul(Y_train.T.float(), output.T.float()).T
            end = time.perf_counter()
            timings.append(end - start)

        # Gem output i dict
        all_outputs.append(output.cpu())

    # Gem alle output i én .pt fil
    torch.save(all_outputs, output_file)
    print(f"All outputs saved in '{output_file}'")
    avg_time_per_sample = sum(timings) / len(timings)
    print(f"Average forward time per sample: {avg_time_per_sample:.6f} s")

def test_model_efficiency(chosen_model, X_test, filters_test, device='cpu'):
    """
    Evaluates computational efficiency of a model on:
    - Inference time (avg per sample)
    - Model size (parameters + file size)
    - Algorithmic order (time scaling with input size)
    """
    # --- Load model ---
    model, output_file = load_model(chosen_model)
    model.to(device)
    model.eval()

    # --- Model size (parameters + file size) ---
    n_params = sum(p.numel() for p in model.parameters())
    torch.save(model.state_dict(), "tmp_model.pt")
    file_size_mb = os.path.getsize("tmp_model.pt") / (1024**2)
    os.remove("tmp_model.pt")

    # --- Runtime and scaling test ---
    times = []
    print(f"\nTesting computational efficiency for {len(data_test)} input sizes...\n")

    for X_test in data_test:
        start = time.perf_counter()
        with torch.no_grad():
            for x in tqdm(X_test, desc=f"Size {tuple(X_test.shape)}"):
                x = x.to(device).unsqueeze(0).float()
                output = model(x)
                if filters_test is not None:
                    output = torch.matmul(filters_test.T.float().to(device), output.T.float()).T
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


save_dir = "Saved Filters"
os.makedirs(save_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#---Load data and split into test and traning data---
data_test, data_train, data_val = load_test_train_data()
X_test = data_test[0]
X_train = data_train[0]
X_val = data_val[0]
filters_test = data_test[1]
filters_train = data_train[1]
filters_val = data_val[1]

#Choose between:
"interpolation"
"regression"
"classification"
"acc"

chosen_model = "interpolation"
print(f"Du har valgt {chosen_model} til evaluering.")
generate_filters(chosen_model, X_test, filters_train) 

