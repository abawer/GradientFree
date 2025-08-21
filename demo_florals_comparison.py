import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import time

# 1. Regression data -------------------------------------------------
X, y = fetch_california_housing(return_X_y=True)
X = X.astype(np.float64)
y = y.astype(np.float64)

# Standardize data
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# simple 80 / 20 split
n = len(X)
split = int(0.8 * n)
X_tr, X_te = X[:split], X[split:]
y_tr, y_te = y[:split], y[split:]

# 2. Hyper-parameters -----------------------------------------------
LAYERS = [128, 64]
ALPHA = 1e-5
BASE_RANK = 64
MIN_RANK = 16
SEED = 42
rng = np.random.default_rng(SEED)

def activation(x): return np.maximum(0, x)  # ReLU for our method
def add_bias(x): return np.hstack([x, np.ones((x.shape[0], 1), dtype=np.float64)])
def g_l(x): return np.tanh(x)  # Non-linear function for FP method

def ridge_solve(A, T, alpha=ALPHA):
    # More stable ridge regression using SVD
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    s = s / (s**2 + alpha)
    return Vt.T @ np.diag(s) @ U.T @ T

# 3. Proposed Method (Low-Rank ALS) Training loop ----------------------------------------------------
start_proposed = time.time()
H_tr_proposed = add_bias(X_tr)
H_te_proposed = add_bias(X_te)

for i, width in enumerate(LAYERS):
    d_in = H_tr_proposed.shape[1]
    
    # Calculate adaptive rank for this layer
    current_rank = max(MIN_RANK, BASE_RANK // (2 ** i))
    
    # rank-1 random target (scalar stretched to `width` dimensions)
    p = rng.standard_normal((width, 1)) / np.sqrt(width)
    T = y_tr.reshape(-1, 1) @ p.T

    # low-rank ALS with adaptive rank
    U = rng.standard_normal((d_in, current_rank)) * np.sqrt(2.0 / d_in)
    Z = H_tr_proposed @ U
    V = ridge_solve(Z, T)
    U = ridge_solve(H_tr_proposed, T @ V.T)
    W = U @ V

    H_tr_proposed = add_bias(activation(H_tr_proposed @ W))
    H_te_proposed = add_bias(activation(H_te_proposed @ W))

# Final scalar regression
FINAL_ALPHA = 1e-3
W_out_proposed = ridge_solve(H_tr_proposed, y_tr.reshape(-1, 1), alpha=FINAL_ALPHA)
y_pred_proposed = H_te_proposed @ W_out_proposed
y_pred_proposed = y_pred_proposed.flatten()

# Convert back to original scale for MSE calculation
y_pred_proposed = scaler_y.inverse_transform(y_pred_proposed.reshape(-1, 1)).flatten()
y_te_orig = scaler_y.inverse_transform(y_te.reshape(-1, 1)).flatten()

mse_proposed = mean_squared_error(y_te_orig, y_pred_proposed)
time_proposed = time.time() - start_proposed

# ELM Implementation for Comparison
start_elm = time.time()
H_tr_elm = add_bias(X_tr)
H_te_elm = add_bias(X_te)

for width in LAYERS:
    d_in = H_tr_elm.shape[1]
    
    # ELM uses completely random weights (no training)
    W_elm = rng.standard_normal((d_in, width)) * np.sqrt(2.0 / d_in)
    
    # Forward pass
    H_tr_elm = add_bias(activation(H_tr_elm @ W_elm))
    H_te_elm = add_bias(activation(H_te_elm @ W_elm))

# Final output layer
W_out_elm = ridge_solve(H_tr_elm, y_tr.reshape(-1, 1), alpha=FINAL_ALPHA)
y_pred_elm = H_te_elm @ W_out_elm
y_pred_elm = y_pred_elm.flatten()

# Convert back to original scale
y_pred_elm = scaler_y.inverse_transform(y_pred_elm.reshape(-1, 1)).flatten()

mse_elm = mean_squared_error(y_te_orig, y_pred_elm)
time_elm = time.time() - start_elm

# Saade's Forward Projection (FP) Method Implementation
start_fp = time.time()
A_tr_fp = X_tr  # Input layer activations (no bias yet)
A_te_fp = X_te

for i, width in enumerate(LAYERS):
    d_in = A_tr_fp.shape[1]
    
    # 1. Create fixed random projection matrices Q_l and U_l
    # Q_l: projects from previous layer dimension to current layer dimension
    Q_l = rng.standard_normal((d_in, width)) / np.sqrt(d_in)
    # U_l: projects from output dimension (1) to current layer dimension
    U_l = rng.standard_normal((1, width)) / np.sqrt(1)
    
    # 2. Generate target potentials Z̃_l = g_l(A_{l-1} Q_l) + g_l(y U_l)
    Z_tilde = g_l(A_tr_fp @ Q_l) + g_l(y_tr.reshape(-1, 1) @ U_l)
    
    # 3. Solve for weights using ridge regression: W_l = (A_{l-1}^T A_{l-1} + λI)^{-1} (A_{l-1}^T Z̃_l)
    W_fp = ridge_solve(A_tr_fp, Z_tilde)
    
    # 4. Forward pass: a_l = f_l(a_{l-1} W_l) - Using tanh as in the paper
    A_tr_fp = np.tanh(A_tr_fp @ W_fp)
    A_te_fp = np.tanh(A_te_fp @ W_fp)

# Final output layer - use ridge regression to map from last hidden layer to output
W_out_fp = ridge_solve(A_tr_fp, y_tr.reshape(-1, 1), alpha=FINAL_ALPHA)
y_pred_fp = A_te_fp @ W_out_fp
y_pred_fp = y_pred_fp.flatten()

# Convert back to original scale
y_pred_fp = scaler_y.inverse_transform(y_pred_fp.reshape(-1, 1)).flatten()

mse_fp = mean_squared_error(y_te_orig, y_pred_fp)
time_fp = time.time() - start_fp

# Backpropagation (BP) Implementation for Comparison
start_bp = time.time()

# Create MLP with similar architecture
bp_model = MLPRegressor(
    hidden_layer_sizes=LAYERS,
    activation='relu',
    solver='adam',
    alpha=1e-4,
    batch_size='auto',
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    random_state=SEED,
    early_stopping=True,
    n_iter_no_change=20,
    validation_fraction=0.1
)

# Train the model
bp_model.fit(X_tr, y_tr)

# Make predictions
y_pred_bp = bp_model.predict(X_te)

# Convert back to original scale
y_pred_bp = scaler_y.inverse_transform(y_pred_bp.reshape(-1, 1)).flatten()

mse_bp = mean_squared_error(y_te_orig, y_pred_bp)
time_bp = time.time() - start_bp

# Print results
print("=" * 70)
print("COMPREHENSIVE COMPARISON RESULTS (SAADE FP METHOD)")
print("=" * 70)
print(f"Proposed Method (Low-Rank ALS):")
print(f"  MSE  = {mse_proposed:.4f}")
print(f"  Time = {time_proposed:.2f} s")
print(f"Saade's FP Method:")
print(f"  MSE  = {mse_fp:.4f}")
print(f"  Time = {time_fp:.2f} s")
print(f"ELM:")
print(f"  MSE  = {mse_elm:.4f}")
print(f"  Time = {time_elm:.2f} s")
print(f"Backpropagation (BP):")
print(f"  MSE  = {mse_bp:.4f}")
print(f"  Time = {time_bp:.2f} s")
print("=" * 70)
print(f"Improvement over Saade FP: {(mse_fp - mse_proposed) / mse_fp * 100:.1f}%")
print(f"Improvement over ELM: {(mse_elm - mse_proposed) / mse_elm * 100:.1f}%")
print(f"Improvement over BP: {(mse_bp - mse_proposed) / mse_bp * 100:.1f}%")
print("=" * 70)
