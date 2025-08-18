import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ----------------------
# 1. Load MNIST
# ----------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

X_train, y_train = next(iter(train_loader))
Y_train = torch.nn.functional.one_hot(y_train, num_classes=10).float()
X_test, y_test = next(iter(test_loader))
Y_test = torch.nn.functional.one_hot(y_test, num_classes=10).float()

# ----------------------
# 2. Define layers
# ----------------------
hidden_dims = [256, 128, 64]
input_dim = X_train.shape[1]

layers = [torch.randn(input_dim if i == 0 else hidden_dims[i-1], hidden_dims[i]) * 0.01
          for i in range(len(hidden_dims))]

# Temporary heads (weights only, linear)
temp_heads = [torch.zeros(h, Y_train.shape[1]) for h in hidden_dims]

# Layer activation function
def act(x): return torch.sigmoid(x)

# ----------------------
# 3. Train temporary heads per layer
# ----------------------
lambda_ridge = 1e-3

H_in = X_train.clone()
residual = Y_train.clone()

for i, layer in enumerate(layers):
    # Forward pass
    H_out = act(H_in @ layer)

    # Linear regression (ridge) for temporary head
    HtH = H_out.T @ H_out + lambda_ridge * torch.eye(H_out.shape[1])
    HtY = H_out.T @ residual
    P_l = torch.linalg.solve(HtH, HtY)
    temp_heads[i] = P_l

    # Update residual
    residual = residual - H_out @ P_l

    # Next layer input
    H_in = H_out

# ----------------------
# 4. Evaluate test accuracy with interim heads
# ----------------------
H_in = X_test.clone()
Y_pred = torch.zeros_like(Y_test)

for layer, P_l in zip(layers, temp_heads):
    H_out = act(H_in @ layer)
    Y_pred += H_out @ P_l
    H_in = H_out

acc = (Y_pred.argmax(dim=1) == y_test).float().mean()
print(f"Test accuracy with interim heads: {acc:.4f}")

# ------------------------------------------------------------------
# 5.  Absorb all heads into ONE dense layer on CONCATENATED features
# ------------------------------------------------------------------
with torch.no_grad():
    # 5-a  collect hidden activations for every layer
    feats = [X_train]                  # input
    h = X_train
    for W in layers:
        h = act(h @ W)
        feats.append(h)                # after layer 0,1,2,…
    H_cat = torch.cat(feats, dim=1)    # N × (d + h1 + … + hL)

    # 5-b  target = greedy logits
    target = torch.zeros_like(Y_train)
    h = X_train
    for W, P in zip(layers, temp_heads):
        h = act(h @ W)
        target += h @ P                # N × C

    # 5-c  ridge regression to predict target from concatenated feats
    reg = lambda_ridge * torch.eye(H_cat.shape[1], device=H_cat.device)
    W_full = torch.linalg.solve(
        H_cat.T @ H_cat + reg,
        H_cat.T @ target)              # (d+h1+…) × C
    b_full = target.mean(0) - (H_cat @ W_full).mean(0)

# ------------------------------------------------------------------
# 6.  Plain MLP that concatenates hidden vectors and applies W_full
# ------------------------------------------------------------------
def plain_mlp(x):
    feats = [x]
    h = x
    for W in layers:
        h = act(h @ W)
        feats.append(h)
    h_cat = torch.cat(feats, dim=1)
    return h_cat @ W_full + b_full

with torch.no_grad():
    logits = plain_mlp(X_test)
    acc_plain = (logits.argmax(dim=1) == y_test).float().mean()
    print(f"Test accuracy with absorbed layer: {acc_plain:.4f}")

# ------------------------------------------------------------------
# 7.  Extreme Learning Machine (ELM) – head only on last hidden layer
# ------------------------------------------------------------------
with torch.no_grad():
    # 7-a  last hidden representation
    h = X_train
    for W in layers:
        h = act(h @ W)

    # 7-b  ridge regression on last hidden → one-hot labels
    reg = lambda_ridge * torch.eye(h.shape[1])
    W_elm = torch.linalg.solve(h.T @ h + reg, h.T @ Y_train)
    b_elm = Y_train.mean(0) - (h @ W_elm).mean(0)

# 7-c  inference
with torch.no_grad():
    h = X_test
    for W in layers:
        h = act(h @ W)
    logits_elm = h @ W_elm + b_elm
    acc_elm = (logits_elm.argmax(dim=1) == y_test).float().mean()

print(f"Test accuracy ELM (last-layer only): {acc_elm:.4f}")
