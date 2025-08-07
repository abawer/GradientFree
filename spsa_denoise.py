import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------
# 1. Synthetic Dataset: sin(x)
# ------------------------------------------------------
torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_data(n=512):
    x = torch.linspace(-np.pi, np.pi, n).unsqueeze(1)
    y = torch.sin(x)
    return x.to(device), y.to(device)

X_train, Y_train = generate_data()

# ------------------------------------------------------
# 2. Simple MLP definition with Xavier init
# ------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 1)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return self.fc3(x)

# ------------------------------------------------------
# 3. Flatten & inject weights
# ------------------------------------------------------
def get_weights(model):
    return torch.cat([p.data.view(-1) for p in model.parameters()])

def set_weights(model, vec):
    i = 0
    for p in model.parameters():
        n = p.numel()
        p.data.copy_(vec[i:i+n].view_as(p))
        i += n

# ------------------------------------------------------
# 4. Loss Function
# ------------------------------------------------------
def compute_loss(model, x, y):
    with torch.no_grad():
        pred = model(x)
        return F.mse_loss(pred, y)

# ------------------------------------------------------
# 5. Initialization
# ------------------------------------------------------
model = MLP().to(device)
W = get_weights(model)
W_dim = W.numel()
latent_dim = 20

print(W_dim, latent_dim)

R = (torch.randn(W_dim, latent_dim, device=device) / latent_dim**0.5)
N = (torch.zeros(latent_dim, device=device))

# ------------------------------------------------------
# 6. Latent Optimization Loop with SPSA
# ------------------------------------------------------
lr_N = 0.01       # learning rate
n_iter = 5000     # iterations
c = 5e-5         # perturbation scale
losses = []

for i in range(n_iter + 1):
    model = MLP().to(device)

    # Generate perturbation vector with ±1 entries
    delta = torch.randint(0, 2, N.shape, device=device).float()
    delta[delta == 0] = -1

    # Perturb latent vector in both directions
    N_plus = N + c * delta
    N_minus = N - c * delta

    W_plus = W + R @ N_plus
    W_minus = W + R @ N_minus

    set_weights(model, W_plus)
    loss_plus = compute_loss(model, X_train, Y_train)

    set_weights(model, W_minus)
    loss_minus = compute_loss(model, X_train, Y_train)

    # SPSA gradient estimate
    grad_est = (loss_plus - loss_minus) / (2 * c) * delta

    # Update latent vector
    N = N - lr_N * grad_est

    if i % 100 == 0:
        print(f"Iter {i:4d} | Loss+: {loss_plus.item():.5f} | Loss-: {loss_minus.item():.5f} | ||N||: {N.norm():.4f}")

    losses.append((loss_plus.item() + loss_minus.item()) / 2)

# ------------------------------------------------------
# 7. Visualize Final Result
# ------------------------------------------------------
model_final = MLP().to(device)
set_weights(model_final, W + R @ N)

with torch.no_grad():
    pred = model_final(X_train)

plt.figure(figsize=(8, 4))
plt.plot(X_train.cpu(), Y_train.cpu(), label="Target", linewidth=2)
plt.plot(X_train.cpu(), pred.cpu(), label="Predicted", linewidth=2)
plt.legend()
plt.title("Final Model after Latent Optimization (SPSA)")
plt.grid()
plt.show()

plt.figure()
plt.plot(losses)
plt.title("Average Loss (SPSA) over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid()
plt.show()
