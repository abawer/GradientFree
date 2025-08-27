# ---------- helpers ----------
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
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
    n_train = int(0.8 * len(X))
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    X_tr, X_te = X[:n_train], X[n_train:]
    y_tr, y_te = y[:n_train], y[n_train:]
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)
    print(f"   done → Train: {X_tr.shape}  Test: {X_te.shape}  ({time.time()-t0:.1f}s)")
    return X_tr, y_tr, X_te, y_te

# ---------- random matrix factories ----------
def rand_mat_old(prev_w, w, rng):
    return rng.standard_normal((prev_w, w)) * np.sqrt(2.0 / prev_w)

def rand_mat_lsh_single(prev_w, w, y_vec, salt):
    """
    Single random matrix *weighted* by smooth Fourier features:
    P = (cos(ω · y) * σ) @ R
    ω and R drawn once; σ scales with label distance.
    """
    rng = np.random.default_rng(hash(salt) & 0xffffffff)

    # 1-D Fourier feature: cos(ω y + φ)   (n × 1)
    ω = rng.standard_normal()
    φ = rng.uniform(0, 2*np.pi)
    f = np.cos(ω * y_vec + φ).reshape(-1, 1)   # (n, 1)

    # single random matrix R   (prev_w × w)
    R = rng.standard_normal((prev_w, w)) * np.sqrt(2.0 / prev_w)

    # weighted average:  (prev_w × w) = (prev_w × w) * mean(f)
    return R * f.mean()          # broadcast scales entire matrix

def rand_mat_lsh(prev_w, w, y_vec, salt):
    y = y_vec.astype(np.float64)
    mu, sig = y.mean(), y.std() + 1e-8
    alpha = 1.0 / (1.0 + np.exp(-(y.mean() - mu) / sig))

    # two seeds from built-in hash
    h = hash((tuple(y), salt))
    seed1 = h & 0xffffffff
    seed2 = (h >> 32) & 0xffffffff

    rng1 = np.random.default_rng(seed1)
    rng2 = np.random.default_rng(seed2)

    A = rng1.standard_normal((prev_w, w)) * np.sqrt(2.0 / prev_w)
    B = rng2.standard_normal((prev_w, w)) * np.sqrt(2.0 / prev_w)
    return alpha * A + (1 - alpha) * B

# ---------- model builder ----------
def build_greedy_model(X, y_real, hidden_widths, alpha, rng, use_label_seeds):
    Ws, bs = [], []
    H = X
    prev_w = X.shape[1]
    for layer_id, w in enumerate(hidden_widths):
        t0 = time.time()
        if use_label_seeds:
            P = rand_mat_lsh_single(prev_w, w, y_real, salt=layer_id)
        else:
            P = rand_mat_old(prev_w, w, rng)
        T = H @ P
        W, b = ridge_with_bias(H, T, alpha)
        H = relu(H @ W + b)
        Ws.append(W)
        bs.append(b)
        prev_w = w
        print(f"Hidden layer {layer_id+1}: {w} units  ({time.time()-t0:.1f}s)")

    # final ridge regression layer
    W_out, b_out = ridge_with_bias(H, y_real.reshape(-1, 1), alpha)
    Ws.append(W_out)
    bs.append(b_out)
    print("=" * 40)
    return Ws, bs

# ---------- predictor ----------
def make_predictor(Ws, bs):
    def predict(X):
        h = X
        for W, b in zip(Ws, bs):
            h = relu(h @ W + b)
        return h.ravel()
    return predict

# ---------- run ----------
def run_once(name, use_label_seeds, hidden_widths=(2000, 1000), alpha=2.0, seed=42):
    rng = np.random.default_rng(seed)
    X_tr, y_tr, X_te, y_te = load_california()
    Ws, bs = build_greedy_model(X_tr, y_tr, hidden_widths, alpha, rng, use_label_seeds)
    predict = make_predictor(Ws, bs)
    rmse_tr = np.sqrt(mean_squared_error(y_tr, predict(X_tr)))
    rmse_te = np.sqrt(mean_squared_error(y_te, predict(X_te)))
    print(f"{name:4s} → Train RMSE : {rmse_tr:.3f}  Test RMSE : {rmse_te:.3f}")
    return rmse_te

if __name__ == "__main__":
    print("Comparing OLD vs NEW label-seeded projections\n")
    old_err = run_once("OLD", use_label_seeds=False)
    new_err = run_once("NEW", use_label_seeds=True)
    print(f"\nΔ(Test RMSE) = {new_err - old_err:+.4f}")
