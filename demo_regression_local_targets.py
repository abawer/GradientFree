# ---------- helpers ----------
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.metrics import mean_squared_error
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
def load_california():
    print("Downloading California Housing …")
    t0 = time.time()
    X, y = fetch_california_housing(return_X_y=True)
    # simple 80/20 split
    n = X.shape[0]
    n_train = int(0.8 * n)
    rng = np.random.default_rng(42)
    idx = rng.permutation(n)
    X, y = X[idx], y[idx]
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
    print(f"   done → Train: {X_tr.shape}  Test: {X_te.shape}  ({time.time()-t0:.1f}s)")
    return X_tr, y_tr, X_te, y_te

# ---------- model ----------
def build_greedy_model(X, y_real, hidden_widths, alpha, rng):
    Ws, bs = [], []
    H = X
    prev_w = X.shape[1]
    for w in hidden_widths:
        t0 = time.time()
        P = rng.standard_normal((prev_w, w)) * np.sqrt(2.0 / prev_w)
        T = H @ P
        W, b = ridge_with_bias(H, T, alpha)
        H = relu(H @ W + b)
        Ws.append(W)
        bs.append(b)
        prev_w = w
        print(f"Hidden layer {len(Ws)}: W {W.shape}, b {b.shape}  ({time.time()-t0:.1f}s)")

    # 2.  Final ridge *regression* layer
    t0 = time.time()
    W_out, b_out = ridge_with_bias(H, y_real.reshape(-1, 1), alpha)
    Ws.append(W_out)
    bs.append(b_out)
    print(f"Regression layer: W {W_out.shape}, b {b_out.shape}  ({time.time()-t0:.1f}s)")
    return Ws, bs

# ---------- predictor ----------
def make_predictor(Ws, bs):
    def predict(X):
        h = X
        for W, b in zip(Ws, bs):
            h = relu(h @ W + b)
        return h.ravel()          # real-valued predictions
    return predict

# ---------- run ----------
def run_demo(hidden_widths=(2000,1000,), alpha=2.0, seed=42):
    rng = np.random.default_rng(seed)
    X_tr, y_tr, X_te, y_te = load_california()
    Ws, bs = build_greedy_model(X_tr, y_tr, hidden_widths, alpha, rng)
    predict = make_predictor(Ws, bs)
    print("=" * 40)

    mse_tr = mean_squared_error(y_tr, predict(X_tr))
    mse_te = mean_squared_error(y_te, predict(X_te))
    print(f"Train RMSE : {np.sqrt(mse_tr):.3f}")
    print(f"Test  RMSE : {np.sqrt(mse_te):.3f}")
    return Ws, bs, predict

if __name__ == "__main__":
    run_demo()
