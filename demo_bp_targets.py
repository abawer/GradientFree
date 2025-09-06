import torch, torch.nn as nn, numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

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
y_test_1d = y_test.ravel()          # 1-D for clean broadcasting

B = 512
train_loader = DataLoader(TensorDataset(torch.from_numpy(X_train),
                                        torch.from_numpy(y_train)),
                          batch_size=B, shuffle=False, drop_last=False)
device = 'cpu'

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

bp_net = BPNet().to(device)
opt = torch.optim.Adam(bp_net.parameters(), lr=3e-3)
crit = nn.MSELoss()

for epoch in range(30):
    bp_net.train()
    mse_sum = 0.
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        loss = crit(bp_net(xb), yb)
        loss.backward()
        opt.step()
        mse_sum += loss.item() * xb.size(0)
print(f'BP train MSE {mse_sum/TR:.4f}')

bp_net.eval()
with torch.no_grad():
    bp_test = crit(bp_net(torch.from_numpy(X_test).to(device)),
                   torch.from_numpy(y_test).to(device)).item()
print(f'BP test MSE {bp_test:.4f}')

# ---------- 3.  capture BP pre-activations -----------------------------------
bp_acts = [[] for _ in range(3)]   # [X, Z1, Z2]
with torch.no_grad():
    for xb, _ in train_loader:
        xb = xb.to(device)
        z1 = bp_net.l1(xb)
        z2 = bp_net.l2(torch.tanh(z1))
        bp_acts[0].append(xb.cpu().numpy())
        bp_acts[1].append(z1.cpu().numpy())
        bp_acts[2].append(z2.cpu().numpy())
bp_acts = [np.vstack(h) for h in bp_acts]

# ---------- 4.  ridge helpers ------------------------------------------------
lam = 8.0
X_np, y_np = X_train, y_train.ravel()

def ridge_bias(X, Y, lam):
    X_ = np.hstack([X, np.ones((X.shape[0], 1), dtype=X.dtype)])
    A = X_.T @ X_ + lam * np.eye(X_.shape[1])
    return np.linalg.solve(A, X_.T @ Y)   # [(d+1) × m]

def mse(a, b):          # safe MSE helper
    return np.mean((a.ravel() - b.ravel()) ** 2)

# ---------- 5.  (a) ridge copy with REAL targets -----------------------------
Wb1 = ridge_bias(X_np, bp_acts[1], lam);  W1, b1 = Wb1[:-1], Wb1[-1]
H1 = np.tanh(X_np @ W1 + b1)
Wb2 = ridge_bias(H1, bp_acts[2], lam);    W2, b2 = Wb2[:-1], Wb2[-1]
H2 = np.tanh(H1 @ W2 + b2)
Wb3 = ridge_bias(H2, y_np.reshape(-1, 1), lam); W3, b3 = Wb3[:-1], Wb3[-1]

# ---------- 5b.  (b) ridge copy with FAKE targets (same μ, σ) ----------------
def fake_like(arr):
    mi, ma = float(arr.min()), float(arr.max())
    return np.clip(np.random.randn(*arr.shape).astype(np.float32), mi, ma)

fake_z1 = fake_like(bp_acts[1])
fake_z2 = fake_like(bp_acts[2])

Wb1_f = ridge_bias(X_np, fake_z1, lam);  W1_f, b1_f = Wb1_f[:-1], Wb1_f[-1]
H1_f = np.tanh(X_np @ W1_f + b1_f)
Wb2_f = ridge_bias(H1_f, fake_z2, lam);  W2_f, b2_f = Wb2_f[:-1], Wb2_f[-1]
H2_f = np.tanh(H1_f @ W2_f + b2_f)
Wb3_f = ridge_bias(H2_f, y_np.reshape(-1, 1), lam); W3_f, b3_f = Wb3_f[:-1], Wb3_f[-1]

# ---------- 6.  evaluation on TEST set -------------------------------------
def forward_np(x, W1, b1, W2, b2, W3, b3):
    h1 = np.tanh(x @ W1 + b1)
    h2 = np.tanh(h1 @ W2 + b2)
    return h2 @ W3 + b3

y_real = forward_np(X_test, W1, b1, W2, b2, W3, b3)
real_mse = mse(y_real, y_test)

y_fake = forward_np(X_test, W1_f, b1_f, W2_f, b2_f, W3_f, b3_f)
fake_mse = mse(y_fake, y_test)

# ---------- 7.  summary ------------------------------------------------------
print('-' * 50)
print(f'BP        test MSE {bp_test:.4f}')
print(f'Real-Z    test MSE {real_mse:.4f}')
print(f'Fake-Z    test MSE {fake_mse:.4f}')
