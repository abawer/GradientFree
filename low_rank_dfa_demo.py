import torch
import torch.nn as nn
import numpy as np
import math
import matplotlib.pyplot as plt

# -----------------------------------------------------
# 1.  Toy data generator
# -----------------------------------------------------
target_func = torch.sin

def generate_data(batch_size=512):
    x = torch.empty(batch_size, 1).uniform_(-np.pi, np.pi)
    y = target_func(x)
    return x, y

# -----------------------------------------------------
# 2.  Network definition  (B is FIXED)
# -----------------------------------------------------
import torch
import torch.nn as nn
import math

class LowRankDFA_MLP(nn.Module):
    """
    DFA network whose feedback matrix B is represented as the outer product
    B = B_L @ B_H^T with fixed, frozen factors B_L (H×r) and B_H (O×r).
    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 rank: int = None):
        super().__init__()
        rank = rank or hidden_dim//4
        # low-rank factors (frozen)
        self.register_buffer('B_L', (2*torch.rand(hidden_dim, rank)-1) / math.sqrt(rank))
        self.register_buffer('B_H', (2*torch.rand(output_dim, rank)-1) / math.sqrt(rank))

        # learnable weights
        self.W1 = nn.Parameter(torch.randn(hidden_dim, input_dim + 1))
        self.W2 = nn.Parameter(torch.randn(output_dim, hidden_dim + 1))

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------
    def forward(self, x):
        x1 = torch.cat([x, torch.ones(x.size(0), 1, device=x.device)], dim=1)
        h  = torch.sigmoid(x1 @ self.W1.t())               # (B, H)
        h1 = torch.cat([h, torch.ones(h.size(0), 1, device=h.device)], dim=1)
        y_pred = h1 @ self.W2.t()                          # (B, O)
        return y_pred, h1

    # ------------------------------------------------------------------
    # local DFA update
    # ------------------------------------------------------------------
    def local_update(self, x, h1, y_pred, y_target, lr):
        e_out = y_pred - y_target                          # (B, O)

        # low-rank feedback: e1 = e_out @ B^T
        tmp = e_out @ self.B_H                             # (B, r)
        e1  = tmp @ self.B_L.t()                           # (B, H)

        # weight updates
        x1 = torch.cat([x, torch.ones(x.size(0), 1, device=x.device)], dim=1)
        self.W1.data -= lr * (e1.t() @ x1) / x1.size(0)
        self.W2.data -= lr * (e_out.t() @ h1) / h1.size(0)

# -----------------------------------------------------
# 3.  Training loop
# -----------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

model = LowRankDFA_MLP(input_dim=1, hidden_dim=64, output_dim=1).to(device)
lr = 0.1
max_epochs = 5000

for epoch in range(max_epochs + 1):
    x_batch, y_batch = generate_data(256)
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)
    y_pred, h1 = model(x_batch)
    loss = nn.MSELoss()(y_pred, y_batch)

    model.local_update(x_batch, h1, y_pred, y_batch, lr)

    if epoch % 100 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.5f}")

# -----------------------------------------------------
# 4.  Sanity plot
# -----------------------------------------------------
model.eval()
with torch.no_grad():
    xs = torch.linspace(-np.pi, np.pi, 400).unsqueeze(1).to(device)
    ys_pred, _ = model(xs)
    ys_true = target_func(xs.cpu())

plt.figure(figsize=(6, 3))
plt.plot(xs.cpu(), ys_pred.cpu(), label='DFA-learned')
plt.plot(xs.cpu(), ys_true, '--', label=f'True {target_func.__name__}(x)')
plt.legend()
plt.title(f'DFA after {max_epochs} epochs (fixed B)')
plt.show()
