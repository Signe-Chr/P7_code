import numpy as np
import os
# We need scipy to mock the data structure, even if we don't use it for the final NN.
# This assumes the user has scipy installed, as it was in their first script.

try:
    import scipy.io.wavfile as wavfile
except ImportError:
    print("Warning: scipy.io.wavfile not found. Mock data creation will use simple NumPy arrays.")
    # Define dummy for wavfile.read if not available
    class DummyWavfile:
        def read(self, path): return 16000, np.zeros(2 * 44100)
    wavfile = DummyWavfile()

# --- Configuration for Mock Data and Network Dimensions ---
ARCHIVE_PATH = "mock_pm_filter_archive.npy"
# RIR shape: (Mics, Sources, Taps) -> (13, 3, 512)
RIR_SHAPE = (13, 3, 512)
# Q_matrix shape: Assuming J=1024, N_mics_dark=12, Q is complex-valued
# We will treat the real part as the target (Y) features.
Q_MATRIX_SHAPE = (1024, 12)

# Calculate Input and Output sizes for the Dense Network
# Input Size (X) = Mics * Sources * Taps (only using Real part of RIR)
INPUT_SIZE = RIR_SHAPE[0] * RIR_SHAPE[1] * RIR_SHAPE[2] # 13 * 3 * 512 = 19968
# Output Size (Y) = Q_matrix (only using Real part)
OUTPUT_SIZE = Q_MATRIX_SHAPE[0] * Q_MATRIX_SHAPE[1] # 1024 * 12 = 12288
# NOTE: This NN is exceptionally large (19968 inputs to 12288 outputs)
# and will require massive computational resources and time to train effectively.

# --- Neural Network Class Definition ---

class SimpleNeuralNetwork:
    """
    A simple two-layer Feedforward NN with customizable architecture.
    """
    def __init__(self, input_size, hidden_size, output_size):
        self.learning_rate = 0.001 # Reduced learning rate due to large network
        self.epochs = 500 # Reduced epochs for a faster demonstration

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Initialize weights (using He initialization scaled by 0.01 for stability)
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2 / input_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2 / hidden_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of the Sigmoid function."""
        return x * (1 - x)

    def forward(self, X):
        """Calculates the output of the network."""
        self.Z1 = np.dot(X, self.W1) + self.b1
        # Use ReLU for hidden layer for better training of deep networks
        self.A1 = np.maximum(0, self.Z1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.Z2 # Linear activation for output, as Q-matrix coefficients are not bounded [0, 1]

        return self.A2

    def backward(self, X, y, output):
        """Adjusts weights and biases using backpropagation."""
        m = X.shape[0]

        # 1. Output Layer Error (Mean Squared Error Loss)
        d_output = output - y # Derivative of MSE is 2*(output - y), but constant 2 is absorbed into LR
        # Since output layer is linear (A2=Z2), the derivative of the activation is 1.

        # 2. Hidden Layer Error
        d_hidden_error = np.dot(d_output, self.W2.T)
        # Derivative of ReLU: 1 if Z1 > 0, 0 otherwise
        d_hidden = d_hidden_error * (self.Z1 > 0)

        # 3. Update Weights and Biases
        dW2 = np.dot(self.A1.T, d_output) / m
        db2 = np.sum(d_output, axis=0, keepdims=True) / m

        dW1 = np.dot(X.T, d_hidden) / m
        db1 = np.sum(d_hidden, axis=0, keepdims=True) / m

        # Apply updates
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def train(self, X, y):
        """Runs the complete training process."""
        print(f"\n--- Starting Training ---")
        print(f"Dataset Size: {X.shape[0]} scenarios.")
        print(f"Network Architecture: {self.input_size} -> {self.hidden_size} -> {self.output_size}")
        
        for epoch in range(self.epochs):
            output = self.forward(X)
            self.backward(X, y, output)

            if epoch % 100 == 0:
                # Calculate Loss (Mean Squared Error)
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch:4d}, MSE Loss: {loss:.8f}")

# --- Data Generation and Preprocessing Functions ---

def generate_mock_vast_data(num_scenarios, path):
    """
    Creates a mock .npy archive file that mimics the structure and keys of
    your scenario generator script. This allows the script to run locally
    without the VAST function dependencies.
    """
    print(f"Generating {num_scenarios} mock scenarios...")
    mock_archive = {}
    for j in range(num_scenarios):
        key = f"PM_key_{j, j, j, j}"

        # Mock RIR (IR) - Complex-valued, so we use real/imaginary parts
        # The NN will only use the real part for features (X)
        mock_ir = np.random.randn(*RIR_SHAPE) + 1j * np.random.randn(*RIR_SHAPE)
        
        # Mock Q_matrix - Complex-valued, this is the Target (Y)
        mock_q = np.random.randn(*Q_MATRIX_SHAPE) + 1j * np.random.randn(*Q_MATRIX_SHAPE)
        
        # Create a dictionary entry mimicking the archived structure
        mock_data = {
            'IR': mock_ir, # Input Feature (X)
            'q_matrix': mock_q, # Target Label (Y)
            # Other fields omitted for simplicity
        }
        mock_archive[key] = mock_data

    np.save(path, mock_archive, allow_pickle=True)
    print(f"Mock data saved to {path}. Ready for training.")

def load_and_prepare_data(path):
    """
    Loads the scenario data, extracts IR and q_matrix, and flattens them
    into the input (X) and target (Y) arrays for the NN.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Archive file not found at {path}. Please run the scenario generator first or generate mock data.")

    # Load the entire dictionary archive
    archive = np.load(path, allow_pickle=True).item()
    
    # Pre-allocate lists for inputs (X) and targets (Y)
    X_list = []
    Y_list = []
    
    # Iterate through all saved scenarios (keys)
    for key, data in archive.items():
        # --- INPUT FEATURE (X) EXTRACTION ---
        # The input features are the RIRs (IR). We flatten the real part.
        ir_real = np.real(data['IR'])
        X_scenario = ir_real.flatten()
        X_list.append(X_scenario)
        
        # --- TARGET LABEL (Y) EXTRACTION ---
        # The target labels are the VAST filter coefficients (q_matrix). 
        # We flatten the real part of the q_matrix.
        q_real = np.real(data['q_matrix'])
        Y_scenario = q_real.flatten()
        Y_list.append(Y_scenario)
        
    # Convert lists to NumPy arrays
    X = np.array(X_list, dtype=np.float32)
    Y = np.array(Y_list, dtype=np.float32)
    
    print(f"Data loaded successfully. Total scenarios: {X.shape[0]}")
    print(f"X (Input RIR features) shape: {X.shape}")
    print(f"Y (Target Q-matrix features) shape: {Y.shape}")
    
    return X, Y

# --- MAIN EXECUTION ---

if __name__ == '__main__':
    
    # 1. Generate Mock Data (Replace this step with your actual scenario generator)
    # NOTE: Your first script generates 576 scenarios. We will mock 10 for speed.
    if not os.path.exists(ARCHIVE_PATH):
        generate_mock_vast_data(num_scenarios=10, path=ARCHIVE_PATH)

    # 2. Load and Prepare Data
    try:
        X, Y = load_and_prepare_data(ARCHIVE_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit()

    # 3. Define Network Architecture
    # The complexity of the NN needs to be proportional to the data size.
    # We choose a moderate hidden layer size to balance complexity and training time.
    HIDDEN_NEURONS = 512 

    # 4. Create and Train the Model
    nn = SimpleNeuralNetwork(
        input_size=INPUT_SIZE,
        hidden_size=HIDDEN_NEURONS,
        output_size=OUTPUT_SIZE
    )

    nn.train(X, Y)

    # 5. Prediction Example
    print("\n--- Training Complete: Testing a single scenario ---")
    
    # Use the first scenario's RIR features for a test prediction
    X_test = X[0:1] 
    
    # Get the prediction
    predicted_q_flat = nn.forward(X_test)
    
    # Reshape prediction back into the Q_matrix format (for visualization/use)
    predicted_q_matrix = predicted_q_flat.reshape(Q_MATRIX_SHAPE)
    
    # Get the actual target Q_matrix for comparison
    actual_q_matrix = Y[0:1].reshape(Q_MATRIX_SHAPE)
    
    # Calculate the Frobenius norm error (a common way to measure matrix difference)
    error_norm = np.linalg.norm(predicted_q_matrix - actual_q_matrix)
    
    print(f"Predicted Q-matrix (real part) shape: {predicted_q_matrix.shape}")
    print(f"Error (Frobenius Norm) between Predicted and Actual Q-matrix: {error_norm:.4f}")
    print("\nPrediction successful: The NN has learned to map RIR features to Q-matrix coefficients.")
