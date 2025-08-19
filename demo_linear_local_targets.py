import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
import time

# -------------------------------------------------
# 1.  Load MNIST
# -------------------------------------------------
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
X = X.astype(np.float32) / 255.0
X_train, X_test = X[:60000], X[60000:]
y_train, y_test = y[:60000], y[60000:]

enc = OneHotEncoder(sparse_output=False, dtype=np.float32)
Y_train = enc.fit_transform(y_train.reshape(-1, 1))   # (60000,10)
Y_test  = enc.transform(y_test.reshape(-1, 1))
C = Y_train.shape[1]

# -------------------------------------------------
# 2.  Settings
# -------------------------------------------------
LAYERS      = [256, 128, 64]
ALPHA       = 1e-4                # ridge
TREE_DEPTH  = 10                  # deeper tree → better targets

def add_bias(X): return np.hstack([X, np.ones((X.shape[0], 1))])

# -------------------------------------------------
# 3.  Layer-wise training
# -------------------------------------------------
H_tr = add_bias(X_train)            # (60000, 785)
H_te = add_bias(X_test)             # (10000, 785)

start = time.time()
np.random.seed(42)

for d_out in LAYERS:
    d_in = H_tr.shape[1]

    # 3-a  random temporary layer
    W_rand = np.random.randn(d_out, d_in) * np.sqrt(2.0 / d_in)
    Z_full = (H_tr @ W_rand.T)          # (60000, d_out)

    μ = np.zeros((C, d_out))          # 10 class means
    for c in range(C):
        mask = (y_train == str(c))
        μ[c] = Z_full[mask].mean(axis=0)

    T = μ[y_train.astype(int)]         # (60000, d_out)

    # 3-c  ridge regression for the layer weights
    clf = Ridge(alpha=ALPHA, fit_intercept=False)
    clf.fit(H_tr, T)
    W = clf.coef_.T                         # (d_in, d_out)

    # 3-d  forward pass
    H_tr = np.tanh(H_tr @ W)
    H_te = np.tanh(H_te @ W)

    print(f"Layer {d_out} done.  shapes: train {H_tr.shape}, test {H_te.shape}")

# -------------------------------------------------
# 4.  Output layer
# -------------------------------------------------
clf_out = Ridge(alpha=ALPHA, fit_intercept=False)
clf_out.fit(add_bias(H_tr), Y_train)
logits = clf_out.predict(add_bias(H_te))
preds  = logits.argmax(axis=1)
acc = accuracy_score(y_test.astype(int), preds)

print(f"\nTest accuracy = {acc:.4f}")
print(f"Total time = {time.time()-start:.1f} s")
