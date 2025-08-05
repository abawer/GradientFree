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
class DFA_MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(hidden_dim, input_dim + 1))
        self.W2 = nn.Parameter(torch.randn(output_dim, hidden_dim + 1))
        self.register_buffer('B', (torch.rand(hidden_dim, output_dim)*2-1)/math.sqrt(hidden_dim))

    def forward(self, x):
        x1 = torch.cat([x, torch.ones(x.size(0), 1, device=x.device)], dim=1)
        h = torch.sigmoid(x1 @ self.W1.t())
        h1 = torch.cat([h, torch.ones(h.size(0), 1, device=h.device)], dim=1)
        y_pred = h1 @ self.W2.t()
        return y_pred, h1

    def local_update(self, x, h1, y_pred, y_target, lr):
        e_out = y_pred - y_target
        e1 = e_out @ self.B.t()
        x1 = torch.cat([x, torch.ones(x.size(0), 1, device=x.device)], dim=1)

        self.W1.data -= lr * (e1.t() @ x1) / x.size(0)
        self.W2.data -= lr * (e_out.t() @ h1) / x.size(0)

# -----------------------------------------------------
# 3.  Training loop
# -----------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

model = DFA_MLP(input_dim=1, hidden_dim=64, output_dim=1).to(device)
lr = 0.1
max_epochs = 5000

for epoch in range(max_epochs + 1):
    x_batch, y_batch = generate_data(128)
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
