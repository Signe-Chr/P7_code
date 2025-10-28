import torch
import torch.nn as nn
import numpy as np
import os
import torch.optim as optim

# --- Configuration (Matches MetadataRegressionNet from previous turn) ---
INPUT_SIZE = 6
FILTER_DIM = 3072
S = 3   # Sources
T = 1024 # Taps

# --- 1. Define the Model Architecture ---
class MetadataRegressionNet(nn.Module):
    """
    A simple Fully Connected network structure for filter coefficient regression.
    This class MUST be defined for PyTorch to load the state_dict correctly.
    """
    def __init__(self, input_size, filter_dim, S, T):
        super().__init__()
        self.S = S
        self.T = T
        
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 256)
        self.fc3 = nn.Linear(256, 1024)
        self.fc_output = nn.Linear(1024, filter_dim)
        self.activation = nn.ReLU() 

    def forward(self, x):
        B = x.size(0)
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        predicted_filters_flat = self.fc_output(x) 
        q = predicted_filters_flat.view(B, self.S, self.T)
        return q

# --- 2. Simulation: Creation and Saving of Two Different Models ---

def create_and_save_models(file1='model_A.pth', file2='model_B.pth'):
    """
    Simulates creating two models, running a small 'training' step on Model B, 
    and saving their state dictionaries.
    """
    print("--- Simulating Model Creation and Saving ---")
    
    # Model A: Initialized weights
    model_a = MetadataRegressionNet(INPUT_SIZE, FILTER_DIM, S, T)
    torch.save(model_a.state_dict(), file1)
    print(f"Saved initial state to: {file1}")

    # Model B: Weights after a simulated training run
    model_b = MetadataRegressionNet(INPUT_SIZE, FILTER_DIM, S, T)
    
    # Simulate one step of training on model B to ensure it differs from model A
    # Setup dummy data and optimizer
    dummy_input = torch.randn(2, INPUT_SIZE)
    dummy_target = torch.randn(2, S, T)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model_b.parameters(), lr=0.01)
    
    # Training step
    optimizer.zero_grad()
    output = model_b(dummy_input)
    loss = criterion(output, dummy_target)
    loss.backward()
    optimizer.step()

    torch.save(model_b.state_dict(), file2)
    print(f"Saved 'trained' state to: {file2}")
    print("------------------------------------------\n")
    return file1, file2


# --- 3. Core Logic: The Comparison Function ---

def compare_models(file_path_a, file_path_b, tolerance=1e-5):
    """
    Loads and compares the state_dicts of two PyTorch models layer-by-layer.
    """
    print("="*50)
    print(f"STARTING COMPARISON: {file_path_a} vs {file_path_b}")
    print("="*50)

    try:
        state_dict_a = torch.load(file_path_a)
        state_dict_b = torch.load(file_path_b)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # 1. Compare Keys (Structure)
    keys_a = set(state_dict_a.keys())
    keys_b = set(state_dict_b.keys())

    if keys_a != keys_b:
        print("\n[FAILED] MODEL STRUCTURE MISMATCH")
        print(f"  Keys only in A: {keys_a - keys_b}")
        print(f"  Keys only in B: {keys_b - keys_a}")
    else:
        print("\n[SUCCESS] Model structures (keys) are identical.")

    # 2. Compare Shapes and Values for Common Keys
    common_keys = sorted(list(keys_a.intersection(keys_b)))
    
    print(f"\nComparing {len(common_keys)} shared parameter tensors...")
    
    all_identical = True

    for key in common_keys:
        tensor_a = state_dict_a[key]
        tensor_b = state_dict_b[key]

        # Check shapes first
        if tensor_a.shape != tensor_b.shape:
            print(f"\n[DIFFERENCE] Shape mismatch for '{key}'")
            print(f"  A Shape: {tensor_a.shape}")
            print(f"  B Shape: {tensor_b.shape}")
            all_identical = False
            continue

        # Check values using torch.allclose
        if not torch.allclose(tensor_a, tensor_b, atol=tolerance):
            all_identical = False
            
            # Calculate metrics for the difference
            diff = torch.abs(tensor_a - tensor_b)
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            
            print(f"\n[DIFFERENCE] Parameter '{key}' differs significantly:")
            print(f"  Tensor Shape: {tensor_a.shape}")
            print(f"  Max Absolute Diff: {max_diff:.8f}")
            print(f"  Mean Absolute Diff: {mean_diff:.8f}")
    
    print("\n" + "="*50)
    if all_identical and keys_a == keys_b:
        print("[FINAL RESULT] The models are structurally and numerically IDENTICAL (within tolerance).")
    elif not all_identical:
        print("[FINAL RESULT] The models have the same structure but DIFFERENT parameter values.")
    else:
        print("[FINAL RESULT] Model comparison complete. See above for details on structural differences.")
    print("="*50 + "\n")


# --- 4. Main Execution Block ---

if __name__ == '__main__':
    # Define file names
    FILE_A = 'mlp_weights.pth' # vægte til dictionary
    FILE_B = 'regression_model_B.pth'

    # 1. Create the dummy files (A is baseline, B is 'trained')
    f_a, f_b = create_and_save_models(FILE_A, FILE_B)

    # 2. Run the comparison
    compare_models(f_a, f_b, tolerance=1e-5)

    # 3. Clean up dummy files
    try:
        os.remove(f_a)
        os.remove(f_b)
        print(f"Cleaned up {f_a} and {f_b}.")
    except OSError:
        pass
