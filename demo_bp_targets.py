# ==========================================================
#  BP  +  Real-Z  +  ES-Z  (single-loss ES)
# ==========================================================
import torch, torch.nn as nn, numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(0)
np.random.seed(0)

# ---------- 1.  data ---------------------------------------------------------
X, y = fetch_california_housing(return_X_y=True, as_frame=False)
X, y = X.astype(np.float32), y.astype(np.float32).reshape(-1, 1)
ss_X, ss_y = StandardScaler(), StandardScaler()
X, y = ss_X.fit_transform(X), ss_y.fit_transform(y)

N = X.shape[0]
idx = np.random.permutation(N)
TR = int(0.8 * N)
X_train, y_train = X[idx[:TR]], y[idx[:TR]]
X_test,  y_test  = X[idx[TR:]], y[idx[TR:]]

B = 512
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train),
                                        torch.from_numpy(y_train)),
                          batch_size=B, shuffle=False, drop_last=False)

# ---------- 2.  BP network ---------------------------------------------------
class BPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(8, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 1)
    def forward(self, x):
        x = torch.tanh(self.l1(x))
        x = torch.tanh(self.l2(x))
        return self.l3(x)

bp_net = BPNet()
crit = nn.MSELoss()
opt  = torch.optim.Adam(bp_net.parameters(), lr=3e-3)

for epoch in range(30):
    bp_net.train()
    mse_sum = 0.
    for xb, yb in train_loader:
        opt.zero_grad()
        loss = crit(bp_net(xb), yb)
        loss.backward()
        opt.step()
        mse_sum += loss.item() * xb.size(0)
bp_train_mse = mse_sum / TR

bp_net.eval()
with torch.no_grad():
    bp_test_mse = crit(bp_net(torch.from_numpy(X_test)),
                       torch.from_numpy(y_test)).item()

# ---------- 3.  capture true pre-activations ---------------------------------
acts = [[] for _ in range(3)]   # [X, z1, z2]
with torch.no_grad():
    for xb, _ in train_loader:
        z1 = bp_net.l1(xb)
        z2 = bp_net.l2(torch.tanh(z1))
        acts[0].append(xb.numpy())
        acts[1].append(z1.numpy())
        acts[2].append(z2.numpy())
acts = [np.vstack(a) for a in acts]          # (N,8) (N,256) (N,128)

# ---------- 4.  ridge helpers ------------------------------------------------
lam = 8.0
X_np, y_np = X_train, y_train.reshape(-1, 1)   # (N,1)  keep 2-D

def ridge_bias(X, Y, lam):
    """X:(N,d)  Y:(N,m)  -> (d+1,m)"""
    X_ = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
    A = X_.T @ X_ + lam * np.eye(X_.shape[1])
    return np.linalg.solve(A, X_.T @ Y)

def mse_np(a, b):
    return np.mean((a - b) ** 2)

def forward_np(x, W1, b1, W2, b2, W3, b3):
    h1 = np.tanh(x @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    return h2 @ W3 + b3

# ---------- 5.  Real-Z (true activations) ------------------------------------
Wb1 = ridge_bias(X_np, acts[1], lam);  W1, b1 = Wb1[:-1], Wb1[-1]
H1  = np.tanh(X_np @ W1 + b1)
Wb2 = ridge_bias(H1,  acts[2], lam);   W2, b2 = Wb2[:-1], Wb2[-1]
H2  = np.tanh(H1 @ W2 + b2)
Wb3 = ridge_bias(H2, y_np, lam);       W3, b3 = Wb3[:-1], Wb3[-1]
real_mse = mse_np(forward_np(X_test, W1, b1, W2, b2, W3, b3), y_test)

# ---------- 6.  ES-Z (single loss) ------------------------------------------
def es_targets(shape, loss_fn, steps=50, noise=0.15, seed=None):
    rng = np.random.RandomState(seed)
    best = rng.randn(*shape).astype(np.float32)
    best_loss = loss_fn(best)
    for step in range(steps):
        cand = best + rng.randn(*shape).astype(np.float32) * noise
        cand_loss = loss_fn(cand)
        if cand_loss < best_loss:
            best, best_loss = cand, cand_loss
            noise *= 1.1
        else:
            noise *= 0.9
        print(f"ES step {step}: best loss {best_loss} noise {noise}")
    return best

# SINGLE loss: full-network MSE
def make_es_loss(X, y, lam):
    def loss(Z):               # Z is (N, hid)
        Wb = ridge_bias(X, Z, lam)
        W, b = Wb[:-1], Wb[-1]
        H = np.tanh(X @ W + b)
        Wb2 = ridge_bias(H, y, lam)
        y_pred = H @ Wb2[:-1] + Wb2[-1]
        return mse_np(y_pred, y)
    return loss

# evolve random Z1
T1 = es_targets(acts[1].shape, make_es_loss(X_np, y_np, lam),
                steps=30, noise=0.15, seed=42)
# build H1, evolve random Z2
Wb1_es = ridge_bias(X_np, T1, lam);  W1_es, b1_es = Wb1_es[:-1], Wb1_es[-1]
H1_es  = np.tanh(X_np @ W1_es + b1_es)
T2 = es_targets(acts[2].shape, make_es_loss(H1_es, y_np, lam),
                steps=100, noise=0.15, seed=43)
# final weights
Wb2_es = ridge_bias(H1_es, T2, lam); W2_es, b2_es = Wb2_es[:-1], Wb2_es[-1]
H2_es  = np.tanh(H1_es @ W2_es + b2_es)
Wb3_es = ridge_bias(H2_es, y_np, lam); W3_es, b3_es = Wb3_es[:-1], Wb3_es[-1]

es_mse = mse_np(forward_np(X_test, W1_es, b1_es,
                            W2_es, b2_es, W3_es, b3_es), y_test)

# ---------- 7.  report -------------------------------------------------------
print('-' * 50)
print(f'BP        test MSE {bp_test_mse:.4f}')
print(f'Real-Z    test MSE {real_mse:.4f}')
print(f'ES-Z      test MSE {es_mse:.4f}')
