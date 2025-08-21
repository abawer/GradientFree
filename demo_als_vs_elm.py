import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
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
LAYERS = [512, 256, 128, 64]
ALPHA  = 1e-10
RANK   = 64
SEED   = 42
rng    = np.random.default_rng(SEED)

def activation(x):  return np.maximum(0, x)
def add_bias(x):    return np.hstack([x, np.ones((x.shape[0], 1), dtype=np.float64)])
def ridge_solve(A, T, alpha=ALPHA):
    AtA = A.T @ A + alpha * np.eye(A.shape[1], dtype=A.dtype)
    return np.linalg.solve(AtA, A.T @ T)

# 3. Proposed Method Training loop ----------------------------------------------------
start_proposed = time.time()
H_tr_proposed = add_bias(X_tr)   # (n_train, 9)
H_te_proposed = add_bias(X_te)   # (n_test , 9)

for width in LAYERS:          # hidden widths
    d_in  = H_tr_proposed.shape[1]
    d_out = width             # neurons in THIS hidden layer

    # rank-1 random target (scalar stretched to `width` dimensions)
    p = rng.standard_normal((width, 1)) / np.sqrt(width)
    T = y_tr.reshape(-1, 1) @ p.T        # (n_samples, width)

    # low-rank ALS
    U = rng.standard_normal((d_in, RANK)) * np.sqrt(2.0 / d_in)
    Z = H_tr_proposed @ U
    V = ridge_solve(Z, T, ALPHA)
    U = ridge_solve(H_tr_proposed, T @ V.T, ALPHA)
    W = U @ V                            # (d_in, width)

    H_tr_proposed = add_bias(activation(H_tr_proposed @ W))
    H_te_proposed = add_bias(activation(H_te_proposed @ W))

# 4. Final scalar regression
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

# Print results
print("=" * 50)
print("COMPARISON RESULTS")
print("=" * 50)
print(f"Proposed Method:")
print(f"  MSE  = {mse_proposed:.4f}")
print(f"  Time = {time_proposed:.2f} s")
print(f"ELM:")
print(f"  MSE  = {mse_elm:.4f}")
print(f"  Time = {time_elm:.2f} s")
print("=" * 50)
print(f"Improvement: {(mse_elm - mse_proposed) / mse_elm * 100:.1f}%")
print("=" * 50)
