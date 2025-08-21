import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import time

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X.astype(np.float64) / 255.0
X_tr, X_te = X[:60000], X[60000:]
y_tr, y_te = y[:60000], y[60000:]

enc = OneHotEncoder(sparse_output=False, dtype=np.float64)
Y_tr = enc.fit_transform(y_tr.reshape(-1, 1))
C = Y_tr.shape[1]

LAYERS = [1024, 512, 256, 128, 64]
ALPHA   = 1e-5
RANK    = 16
SEED    = 42
rng = np.random.default_rng(SEED)

def activation(X):
    return np.tanh(X)

def add_bias(X):
    return np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float64)])

def ridge_solve(A, T, alpha):
    AtA = A.T @ A + alpha * np.eye(A.shape[1], dtype=A.dtype)
    return np.linalg.solve(AtA, A.T @ T)

# --------------------------------------------------
start = time.time()
H_tr = add_bias(X_tr)   # (60000, 785)
H_te = add_bias(X_te)   # (10000, 785)

for d_out in LAYERS:
    d_in = H_tr.shape[1]

    # --- 1. Build a d_out-dimensional target T ---------------------------
    P = rng.standard_normal((C, d_out)) / np.sqrt(C)
    T = Y_tr @ P                                   # (n_samples, d_out)

    # --- 2. Low-rank ALS:  W = U V^T  with U∈ℝ^{d_in×RANK}, V∈ℝ^{d_out×RANK}
    U = rng.standard_normal((d_in, RANK)) * np.sqrt(2.0 / d_in)

    # step-1: solve for V   (RANK, d_out)
    Z = H_tr @ U                            # (n_samples, RANK)
    V = ridge_solve(Z, T, ALPHA)           # (RANK, d_out)

    # step-2: solve for U   (d_in, RANK)
    U = ridge_solve(H_tr, T @ V.T, ALPHA)   # (d_in, RANK)

    # final rank-RANK weight
    W = U @ V                                    # (d_in, d_out)  # Removed .T from V

    # forward pass
    H_tr = add_bias(activation(H_tr @ W))
    H_te = add_bias(activation(H_te @ W))

    print(f"Layer {d_out} done.  shapes: {H_tr.shape}, {H_te.shape}")

# output layer
W_out = ridge_solve(H_tr, Y_tr, ALPHA)
logits = H_te @ W_out
preds  = logits.argmax(1)
print(f"\nTest accuracy = {accuracy_score(y_te.astype(int), preds):.4f}")
print(f"Total time = {time.time()-start:.1f} s")
