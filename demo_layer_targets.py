# ---------- helpers ----------
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score
import time

relu = lambda x: np.maximum(0, x)

def ridge_with_bias(A, b, alpha):
    A_ext = np.hstack([A, np.ones((A.shape[0], 1))])
    t0 = time.time()
    Wb = np.linalg.solve(A_ext.T @ A_ext + alpha * np.eye(A_ext.shape[1]),
                         A_ext.T @ b)
    W, b = Wb[:-1], Wb[-1]
    print(f"   ridge → W {W.shape}, b {b.shape}  ({time.time()-t0:.1f}s)")
    return W, b

# ---------- data ----------
def load_fmnist():
    print("Downloading FMNIST …")
    t0 = time.time()
    Xy = fetch_openml('Fashion-MNIST', version=1, as_frame=False)
    X, y = Xy.data.astype('float32'), Xy.target.astype('int')
    n_train = 60_000
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
    ohe = OneHotEncoder(sparse_output=False).fit(y_tr.reshape(-1, 1))
    Y_tr = ohe.transform(y_tr.reshape(-1, 1))
    Y_te = ohe.transform(y_te.reshape(-1, 1))
    print(f"   done → Train: {X_tr.shape}  Test: {X_te.shape}  ({time.time()-t0:.1f}s)")
    return X_tr, y_tr, X_te, y_te

# ---------- model ----------
def build_greedy_model(X, y_int, hidden_widths, alpha, rng, n_class=10):
    Ws, bs = [], []
    H = X
    prev_w = X.shape[1]
    for w in hidden_widths:
        t0 = time.time()
        P = rng.standard_normal((n_class, prev_w, w)) * np.sqrt(2.0 / prev_w)
        T = np.empty((H.shape[0], w))
        for c in range(n_class):
            mask = y_int == c
            T[mask] = H[mask] @ P[c]
        W, b = ridge_with_bias(H, T, alpha)
        H = relu(H @ W + b)
        Ws.append(W)
        bs.append(b)
        prev_w = w
        print(f"Hidden layer {len(Ws)}: W {W.shape}, b {b.shape}  ({time.time()-t0:.1f}s)")

    # logit layer
    t0 = time.time()
    W_out, b_out = ridge_with_bias(H, np.eye(n_class)[y_int], alpha)
    Ws.append(W_out)
    bs.append(b_out)
    print(f"Logit layer: W {W_out.shape}, b {b_out.shape}  ({time.time()-t0:.1f}s)")
    return Ws, bs

# ---------- predictor ----------
def make_predictor(Ws, bs):
    def predict(X):
        h = X
        for W, b in zip(Ws, bs):
            h = relu(h @ W + b)
        return h.argmax(1)
    return predict

# ---------- run ----------
def run_demo(hidden_widths=(1000, 1000, 1000), alpha=1.0, seed=42):
    rng = np.random.default_rng(seed)
    X_tr, y_tr, X_te, y_te = load_fmnist()
    Ws, bs = build_greedy_model(X_tr, y_tr, hidden_widths, alpha, rng)
    predict = make_predictor(Ws, bs)
    print("=" * 40)
    print(f"Train accuracy : {accuracy_score(y_tr, predict(X_tr))*100:5.2f}%")
    print(f"Test  accuracy : {accuracy_score(y_te, predict(X_te))*100:5.2f}%")
    return Ws, bs, predict

if __name__ == "__main__":
    run_demo()
