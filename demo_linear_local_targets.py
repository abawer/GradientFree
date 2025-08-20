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
SEED    = 42
rng = np.random.default_rng(SEED)

def add_bias(X):
    return np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float64)])

def ridge_solve(A, T, alpha):
    AtA = A.T @ A + alpha * np.eye(A.shape[1], dtype=A.dtype)
    return np.linalg.solve(AtA, A.T @ T)

H_tr = add_bias(X_tr)
H_te = add_bias(X_te)

start = time.time()
for d_out in LAYERS:
    # class-conditioned targets
    W_y = rng.standard_normal((C, d_out), dtype=np.float64) / np.sqrt(C)
    T   = Y_tr.astype(np.float64) @ W_y
    # layer weights
    W = ridge_solve(H_tr, T, ALPHA * d_out)
    H_tr = np.tanh(H_tr @ W)
    H_te = np.tanh(H_te @ W)
    print(f"Layer {d_out} done.  shapes: {H_tr.shape}, {H_te.shape}")

# output layer
W_out = ridge_solve(add_bias(H_tr), Y_tr, ALPHA * C)
logits = add_bias(H_te) @ W_out
preds  = logits.argmax(1)
print(f"\nTest accuracy = {accuracy_score(y_te.astype(int), preds):.4f}")
print(f"Total time = {time.time()-start:.1f} s")
