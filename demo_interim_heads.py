# ======================================================================
# 0.  Imports and MNIST (same as before)
# ======================================================================
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
train_loader = DataLoader(
    datasets.MNIST('./data', train=True,  download=True, transform=transform),
    batch_size=len(datasets.MNIST('./data', train=True))
)
test_loader = DataLoader(
    datasets.MNIST('./data', train=False, download=True, transform=transform),
    batch_size=len(datasets.MNIST('./data', train=False))
)

X_train, y_train = next(iter(train_loader))
X_test,  y_test  = next(iter(test_loader))
Y_train = torch.nn.functional.one_hot(y_train, 10).float()
Y_test  = torch.nn.functional.one_hot(y_test,  10).float()

# ======================================================================
# 1.  Network definition
# ======================================================================
hidden_dims = [256, 128, 64]
layers = [torch.randn(X_train.shape[1] if i == 0 else hidden_dims[i-1],
                      hidden_dims[i]) * 0.01
          for i in range(len(hidden_dims))]
act = torch.sigmoid
lambda_ridge = 1e-3
k = 64   # shared low-rank

# ======================================================================
# 2.  Hidden states (train & test)
# ======================================================================
def hidden_states(x):
    feats = [x]
    h = x
    for W in layers:
        h = act(h @ W)
        feats.append(h)
    return feats

H_train = hidden_states(X_train)
H_test  = hidden_states(X_test)
H_cat_train = torch.cat(H_train, dim=1)   # N × Σh_i

# ======================================================================
# 3.  Exact greedy logits (interim heads)
# ======================================================================
residual = Y_train.clone()
logits_train = torch.zeros_like(Y_train)
heads = []   # list of h_i×10 matrices

for h_i in H_train[1:]:        # skip input
    reg_i = lambda_ridge * torch.eye(h_i.shape[1])
    P_i = torch.linalg.solve(h_i.T @ h_i + reg_i, h_i.T @ residual)
    heads.append(P_i)
    logits_train += h_i @ P_i
    residual -= h_i @ P_i

# test logits
logits_test = torch.zeros_like(Y_test)
for h_i, P in zip(hidden_states(X_test)[1:], heads):
    logits_test += h_i @ P
acc_interim = (logits_test.argmax(1) == y_test).float().mean()

# ======================================================================
# 4.  Low-rank absorption into ONE final layer
# ======================================================================
with torch.no_grad():
    # Ridge on concatenated hidden states
    reg = lambda_ridge * torch.eye(H_cat_train.shape[1])
    W_full = torch.linalg.solve(H_cat_train.T @ H_cat_train + reg,
                                H_cat_train.T @ logits_train)  # Σh_i×10

    # SVD truncation to rank k
    U, S, Vt = torch.linalg.svd(W_full, full_matrices=False)
    W_low = U[:, :k] @ torch.diag(S[:k])   # Σh_i×k
    B_low = Vt[:k]                        # k×10
    b_final = logits_train.mean(0) - (H_cat_train @ W_full).mean(0)

def low_rank_mlp(x):
    h = torch.cat(hidden_states(x), dim=1)   # N×Σh_i
    return (h @ W_low) @ B_low + b_final     # N×k @ k×10 → N×10

acc_low = (low_rank_mlp(X_test).argmax(1) == y_test).float().mean()

# ======================================================================
# 5.  ELM baseline (ridge on last hidden layer only)
# ======================================================================
h_last_train = H_train[-1]
h_last_test  = H_test[-1]

with torch.no_grad():
    reg = lambda_ridge * torch.eye(h_last_train.shape[1])
    W_elm = torch.linalg.solve(h_last_train.T @ h_last_train + reg,
                               h_last_train.T @ Y_train)
    logits_elm = h_last_test @ W_elm
    acc_elm = (logits_elm.argmax(1) == y_test).float().mean()

# ======================================================================
# 6.  Print
# ======================================================================
print(f"Test accuracy interim heads:   {acc_interim:.4f}")
print(f"Test accuracy low-rank MLP:    {acc_low:.4f}")
print(f"Test accuracy ELM:             {acc_elm:.4f}")
