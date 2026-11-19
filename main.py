import numpy as np
import pandas as pd
import re
import warnings

# --- 1. Configuration & Parameters ---
# File path for the Unweighted Supermatrix (CSV version from upload)
# Use the path accessible by the environment
csv_file_path = 'Un_weighted_superM.csv' # Adjust if needed

# --- Parameters for Bayesian Sampling ---
NUM_SAMPLES = 50000  # Number of samples to draw from the BNN parameter distributions

# --- Define Standard Deviations for Priors ---
# These represent the assumed uncertainty around the mean values.
W1_PRIOR_STD = 0.1 # Uncertainty around FANP W1 values (Keep as before, or tune)
B1_PRIOR_STD = 0.1 # Uncertainty around FANP B1 values (Keep as before, or tune)

# For W2, use the transpose of W1_mean as the PRIOR MEAN.
# *** Use EVEN TIGHTER std dev based on user request to tune ***
W2_PRIOR_STD = 0.01 # Tighter value (e.g., 0.05, 0.02) - EXPERIMENT HERE

# For B2, keep a less informative prior (mean 0) but with tighter std dev.
B2_PRIOR_MEAN = 0.0
B2_PRIOR_STD = 0.01 # Tighter value - EXPERIMENT HERE

# --- Activation Functions ---
def tanh(x):
    """ Hyperbolic tangent activation function. """
    return np.tanh(x)

# Define the Softmax function for output layer
def softmax(x):
    """ Compute softmax values for each set of scores in x. """
    # Subtract max for numerical stability (prevents overflow)
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

# --- Load Data from CSV ---
try:
    df = pd.read_csv(csv_file_path, header=0)
    print(f"Successfully loaded data from {csv_file_path}")
except FileNotFoundError:
    print(f"Error: File not found at {csv_file_path}")
    exit()
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit()

# --- Data Extraction and Preparation (Using DataFrame) ---
# (Same data extraction logic as the previous correct version)
expected_component_col = 'Unnamed: 1'
expected_first_subcrit_col = 'C1S1'
expected_last_subcrit_col = 'C5S3'

if not all(col in df.columns for col in [expected_component_col, expected_first_subcrit_col, expected_last_subcrit_col]):
    print("Error: Expected columns not found in CSV.")
    exit()

component_col_idx = df.columns.get_loc(expected_component_col)
first_subcrit_col_idx = df.columns.get_loc(expected_first_subcrit_col)
last_subcrit_col_idx = df.columns.get_loc(expected_last_subcrit_col)

n_components = 11
component_names = df.iloc[0:n_components, component_col_idx].values.tolist()

n_subcriteria = 15
sub_criteria_names = df.columns[first_subcrit_col_idx : last_subcrit_col_idx + 1].tolist()

if len(component_names) != n_components or len(sub_criteria_names) != n_subcriteria:
     print("Error: Incorrect number of components or sub-criteria extracted.")
     exit()

print(f"\nComponents (Inputs/Outputs): {component_names}")
print(f"Sub-Criteria (Hidden Layer): {sub_criteria_names}")

try:
    W1_mean_raw = df.iloc[0:n_components, first_subcrit_col_idx : last_subcrit_col_idx + 1]
    if np.isnan(W1_mean_raw).any():
        print("Warning: NaN values found in W1 means. Replacing NaNs with 0.")
        W1_mean_raw = np.nan_to_num(W1_mean_raw)
    if W1_mean_raw.shape != (n_components, n_subcriteria): raise ValueError("W1 shape")
    W1_prior_mean = W1_mean_raw
    print(f"\nShape of W1 Prior Mean (Input-to-Hidden): {W1_prior_mean.shape}")
except Exception as e:
    print(f"Error extracting W1 means: {e}")
    exit()

sub_criteria_final_weights = {
    'C1S1': 0.4290, 'C1S2': 0.3157, 'C1S3': 0.2553, 'C2S1': 0.5730, 'C2S2': 0.4270,
    'C3S1': 0.4026, 'C3S2': 0.2995, 'C3S3': 0.2978, 'C4S1': 0.3280, 'C4S2': 0.2740,
    'C4S3': 0.2106, 'C4S4': 0.1875, 'C5S1': 0.4478, 'C5S2': 0.3331, 'C5S3': 0.2191
}

try:
    B1_prior_mean = np.array([sub_criteria_final_weights[name] for name in sub_criteria_names])
    if B1_prior_mean.shape != (n_subcriteria,): raise ValueError("B1 shape")
    print(f"Shape of B1 Prior Mean (Hidden Layer Biases): {B1_prior_mean.shape}")
except KeyError as e:
    print(f"Error: Sub-criterion name mismatch. '{e}'.")
    exit()

# --- 2. BNN Forward Pass Definition (Single Sample) ---
def predict_BNN_sample(input_vector,
                       w1_mean, b1_mean, w1_std, b1_std, # Layer 1 params
                       b2_mean_val, w2_std, b2_std,        # Layer 2 params
                       hidden_activation=tanh, # *** USING tanh AS REQUESTED ***
                       output_activation=softmax):
    """
    Performs a single forward pass using weights/biases sampled from Normal priors.
    Uses W1_mean.T as the mean for the W2 prior distribution.
    Applies specified hidden (default tanh) and output (default Softmax) activations.
    """
    n_in, n_hid = w1_mean.shape
    n_out = len(input_vector)

    # Sample parameters for Layer 1
    w1_sample = np.random.normal(loc=w1_mean, scale=w1_std, size=w1_mean.shape)
    b1_sample = np.random.normal(loc=b1_mean, scale=b1_std, size=b1_mean.shape)

    # Define parameters for Layer 2
    w2_mean_matrix = w1_mean.T
    if w2_mean_matrix.shape != (n_hid, n_out):
         raise ValueError(f"W2 prior mean shape mismatch: expected ({n_hid}, {n_out}), got {w2_mean_matrix.shape}")
    b2_mean_vector = np.full((n_out,), b2_mean_val)

    # Sample parameters for Layer 2
    w2_sample = np.random.normal(loc=w2_mean_matrix, scale=w2_std, size=(n_hid, n_out))
    b2_sample = np.random.normal(loc=b2_mean_vector, scale=b2_std, size=(n_out,))

    # Perform the forward calculation
    hidden_input = np.dot(input_vector, w1_sample) + b1_sample
    hidden_output = hidden_activation(hidden_input) # *** Apply tanh (or passed activation)
    output_input = np.dot(hidden_output, w2_sample) + b2_sample
    final_output = output_activation(output_input) # Apply softmax
    return final_output

# --- 3. Execution: Sample from the BNN Priors ---
input_vector = np.ones(n_components)
print(f"\nInput Vector (shape {input_vector.shape}):\n{input_vector}")
print(f"\n--- Starting BNN Sampling ({NUM_SAMPLES} samples) ---")
print(f"Using **Tanh** hidden activation and Softmax output.")
print(f"Using W1.T as W2 prior mean.")
print(f"Using W1/B1 prior std dev: {W1_PRIOR_STD:.2f}")
print(f"Using W2/B2 prior std dev: {W2_PRIOR_STD:.2f} / {B2_PRIOR_STD:.2f} (TIGHTENED)")

all_predictions = []
for i in range(NUM_SAMPLES):
    prediction = predict_BNN_sample(
        input_vector,
        W1_prior_mean, B1_prior_mean, W1_PRIOR_STD, B1_PRIOR_STD,
        B2_PRIOR_MEAN, W2_PRIOR_STD, B2_PRIOR_STD,
        hidden_activation=tanh # Explicitly pass tanh if not changing default
        # output_activation defaults to softmax
    )
    all_predictions.append(prediction)
    if (i + 1) % (NUM_SAMPLES // 10) == 0: print(f"  Completed sample {i+1}/{NUM_SAMPLES}")

all_predictions = np.array(all_predictions)
print("--- Sampling Complete ---")

# --- 4. Results: Analyze the Posterior Predictive Distribution ---
mean_criticalities = np.mean(all_predictions, axis=0)
std_dev_criticalities = np.std(all_predictions, axis=0)

print("\n--- BNN Criticality Results (Tanh Hidden, Tight Prob Transpose Priors & Softmax) ---")
print("Showing Mean and Standard Deviation of the predicted RELATIVE criticality distribution")

output_results = {
    'Component': component_names,
    'Mean_Relative_Crit': mean_criticalities,
    'StdDev_Relative_Crit': std_dev_criticalities
}
results_df = pd.DataFrame(output_results)
print(f"\nSum of Mean Relative Criticalities: {np.sum(mean_criticalities):.5f}")
print(results_df.to_string(index=False, float_format="%.5f"))

# --- Optional: Comparison with Paper's Values ---
target_criticalities_paper = {
    'A1': 0.06418, 'A2': 0.04086, 'A3': 0.02003, 'A4': 0.02730,
    'A5': 0.02468, 'A6': 0.02701, 'A7': 0.01883, 'A8': 0.01873,
    'A9': 0.02725, 'A10': 0.03869, 'A11': 0.02577
}

try:
    target_criticalities_vector = np.array([target_criticalities_paper[name] for name in component_names])
    results_df['Target_Paper'] = target_criticalities_vector
    results_df['Diff_Mean_vs_Target'] = results_df['Mean_Relative_Crit'] - results_df['Target_Paper']
    print("\n--- Comparison with Paper's Values (for context ONLY) ---")
    print(results_df.to_string(index=False, float_format="%.5f"))
    mae_mean_vs_target = np.mean(np.abs(results_df['Diff_Mean_vs_Target']))
    print(f"\nMean Absolute Error (Mean Prediction vs Paper Target): {mae_mean_vs_target:.5f}")
    print("(Note: Low MAE is NOT the goal here; the goal is a new method with uncertainty.)")
except KeyError as e:
    print(f"\nWarning: Could not compare with paper targets. Component name '{e}'.")

print("\n--- Important Notes ---")
print("1. BNN using **prior sampling**, **probabilistic transpose** for W2 mean.")
print("2. Hidden activation is **Tanh**.")
print("3. Output uses **Softmax**.")
print("4. **Tightened** prior standard deviations for W2/B2 used (e.g., 0.05).")
print("5. Results show mean and std dev of **relative** criticalities.")
print("6. Further tuning of PRIOR STANDARD DEVIATIONS may be needed.")