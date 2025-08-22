"""
Fashion-MNIST single-run comparison
- Proposed (low-rank ALS)
- Forward Propagation (Saade et al.)
- Plain ELM
- ELM-X (orthogonal full-width + LWLR head)
"""

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import accuracy_score
import time

# -----------------------------------------------------------
# 1. Data utilities
# -----------------------------------------------------------
def load_fmnist():
    X, y = fetch_openml('Fashion-MNIST', version=1, return_X_y=True)
    X = X.astype(np.float32) / 255.0
    y = y.astype(int)
    lb = LabelBinarizer()
    y_onehot = lb.fit_transform(y)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_onehot, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)
    return X_tr, X_te, y_tr, y_te, lb

# -----------------------------------------------------------
# 2. Helper functions
# -----------------------------------------------------------
def add_bias(x):
    return np.hstack([x, np.ones((x.shape[0], 1), dtype=x.dtype)])

def ridge_solve(A, T, alpha=1e-5):
    AtA = A.T @ A + alpha * np.eye(A.shape[1], dtype=A.dtype)
    return np.linalg.solve(AtA, A.T @ T)

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

# -----------------------------------------------------------
# 3. Algorithms
# -----------------------------------------------------------
# 3.1 Low-rank ALS (memory-safe)
def proposed_method_classification(
        X_tr, X_te, y_tr, layers,
        alpha=1e-5, seed=42):

    rng = np.random.default_rng(seed)
    act = lambda x: np.maximum(0, x)

    H_tr = add_bias(X_tr)
    H_te = add_bias(X_te)

    C = y_tr.shape[1]
    rank = 512                     # large enough sketch
    min_rank = C

    for width in layers:
        d_in = H_tr.shape[1]
        r = max(C, int(np.sqrt(width)))        # identical rule for factors

        # low-rank target factors
        P_l = rng.standard_normal((C, r)) * np.sqrt(2.0 / C)
        P_r = rng.standard_normal((r, width)) * np.sqrt(2.0 / r)
        Z = H_tr @ rng.standard_normal((d_in, r)) * np.sqrt(2.0 / d_in)

        V = ridge_solve(Z, y_tr @ P_l, alpha) @ P_r
        tmp = P_r @ V.T
        U = ridge_solve(H_tr, y_tr @ P_l @ tmp, alpha)
        W = U @ V

        H_tr = add_bias(act(H_tr @ W))
        H_te = add_bias(act(H_te @ W))

        rank = max(rank // 2, min_rank)

    W_out = ridge_solve(H_tr, y_tr, alpha=1e-3)
    return softmax(H_te @ W_out)

# 3.2 Forward Propagation (Saade et al.)
def forward_propagation_classification(
        X_tr, X_te, y_tr, layers,
        alpha=1e-5, seed=42):

    rng = np.random.default_rng(seed)
    act = lambda x: np.maximum(0, x)

    H_tr = add_bias(X_tr)
    H_te = add_bias(X_te)

    C = y_tr.shape[1]

    for width in layers:
        d_in = H_tr.shape[1]

        Q = rng.normal(0, 1 / np.sqrt(d_in), (d_in, width))
        U = rng.normal(0, 1 / np.sqrt(C), (C, width))

        Z_tilde = np.sign(H_tr @ Q) + np.sign(y_tr @ U)
        W_fp = ridge_solve(H_tr, Z_tilde, alpha)

        H_tr = add_bias(act(H_tr @ W_fp))
        H_te = add_bias(act(H_te @ W_fp))

    W_out = ridge_solve(H_tr, y_tr, alpha=1e-3)
    return softmax(H_te @ W_out)

# 3.3 Plain ELM
def elm_method_classification(
        X_tr, X_te, y_tr, layers,
        alpha=1e-5, seed=42):

    rng = np.random.default_rng(seed)
    act = lambda x: np.maximum(0, x)

    H_tr = add_bias(X_tr)
    H_te = add_bias(X_te)

    for width in layers:
        d_in = H_tr.shape[1]
        W = rng.standard_normal((d_in, width)) * np.sqrt(2.0 / d_in)
        H_tr = add_bias(act(H_tr @ W))
        H_te = add_bias(act(H_te @ W))

    W_out = ridge_solve(H_tr, y_tr, alpha=1e-3)
    return softmax(H_te @ W_out)

# 3.4 ELM-X (orthogonal full-width + LWLR head)
def elmx_classification(
        X_tr, X_te, y_tr, layers,
        alpha=1e-5, seed=42):

    rng = np.random.default_rng(seed)
    act = lambda x: np.maximum(0, x)

    H_tr = add_bias(X_tr)
    H_te = add_bias(X_te)

    C = y_tr.shape[1]

    for width in layers:
        d_in = H_tr.shape[1]

        # 1. Orthogonal random features
        Q_raw = rng.standard_normal((d_in, width))
        Q, _ = np.linalg.qr(Q_raw)
        Q *= np.sqrt(2.0 * width / d_in)             # He variance
        Z_tr = H_tr @ Q                              # (n_samples, width)
        Z_te = H_te @ Q                              # (n_samples, width)

        # 2. LWLR head
        W = ridge_solve(Z_tr, y_tr, alpha)           # (width, C)

        # 3. Variance-preserving scaling
        W *= np.sqrt(2.0 / (W**2).mean())

        # forward pass
        H_tr = add_bias(act(Z_tr @ W))
        H_te = add_bias(act(Z_te @ W))

    # final ridge head
    W_out = ridge_solve(H_tr, y_tr, alpha=1e-3)
    return softmax(H_te @ W_out)

# -----------------------------------------------------------
# 4. Single-run comparison
# -----------------------------------------------------------
def compare_methods_fmnist():
    X_tr, X_te, y_tr, y_te, _ = load_fmnist()
    layers = [1000, 1000, 200]
    seed = 42

    methods = {
        "Proposed (Low-Rank ALS)": proposed_method_classification,
        "Forward Propagation":      forward_propagation_classification,
        "ELM":                      elm_method_classification,
        "ELM-X":                    elmx_classification,
    }

    results = {}
    for name, method in methods.items():
        t0 = time.time()
        try:
            probs_te = method(X_tr, X_te, y_tr, layers, seed=seed)
            acc_te = accuracy_score(np.argmax(y_te, axis=1),
                                    np.argmax(probs_te, axis=1))
            dt = time.time() - t0
            results[name] = {"test": acc_te, "time": dt}
            print(f"{name:20s}: test={acc_te:.4f}  ({dt:.1f}s)")
        except Exception as e:
            print(f"{name:20s}: FAILED — {e}")
            results[name] = {"test": np.nan, "time": np.nan}
    return results

# -----------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Fashion-MNIST single-run comparison")
    print("=" * 60)
    compare_methods_fmnist()
