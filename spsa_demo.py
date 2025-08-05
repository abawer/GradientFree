import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------------
# 1. Toy data generator
# -----------------------------------------------------
target_func = torch.sin  # or torch.cos
def generate_data(batch_size=512):
    x = torch.empty(batch_size, 1).uniform_(-np.pi, np.pi)
    y = target_func(x)
    return x, y

# -----------------------------------------------------
# 2. Network with frozen noisy weights and full correction matrices
# -----------------------------------------------------
class SPSA_CorrectedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        # Fixed noisy weights (frozen)
        self.W1_noisy = nn.Parameter(torch.randn(hidden_dim, input_dim+1), requires_grad=False)
        self.W2_noisy = nn.Parameter(torch.randn(output_dim, hidden_dim+1), requires_grad=False)

        # Full Correction Matrices initialized as Identity
        self.C1 = nn.Parameter(torch.eye(hidden_dim))
        self.C2 = nn.Parameter(torch.eye(output_dim))

    def forward(self, x):
        x1 = torch.cat([x, torch.ones(x.size(0), 1, device=x.device)], dim=1)

        W1_corr = self.C1 @ self.W1_noisy
        h = torch.sigmoid(x1 @ W1_corr.t())

        h1 = torch.cat([h, torch.ones(h.size(0), 1, device=h.device)], dim=1)

        W2_corr = self.C2 @ self.W2_noisy
        y_pred = h1 @ W2_corr.t()
        return y_pred

# -----------------------------------------------------
# 3. SPSA update for correction matrices
# -----------------------------------------------------
def spsa_update(model, x, y_target, lr, epsilon=1e-4):
    criterion = nn.MSELoss()
    with torch.no_grad():
        def compute_loss(C1, C2):
            # Temporarily swap correction matrices
            old_C1, old_C2 = model.C1.clone(), model.C2.clone()
            model.C1.copy_(C1)
            model.C2.copy_(C2)
            y_pred = model(x)
            loss = criterion(y_pred, y_target).item()
            # Restore old values
            model.C1.copy_(old_C1)
            model.C2.copy_(old_C2)
            return loss

        # SPSA step for C1
        delta_C1 = torch.randint(0, 2, model.C1.shape, device=model.C1.device, dtype=torch.float32)*2 - 1  # ±1
        C1_plus  = model.C1 + epsilon * delta_C1
        C1_minus = model.C1 - epsilon * delta_C1

        loss_plus = compute_loss(C1_plus, model.C2)
        loss_minus = compute_loss(C1_minus, model.C2)
        grad_C1_est = (loss_plus - loss_minus) / (2 * epsilon) * delta_C1

        # SPSA step for C2
        delta_C2 = torch.randint(0, 2, model.C2.shape, device=model.C2.device, dtype=torch.float32)*2 - 1  # ±1
        C2_plus  = model.C2 + epsilon * delta_C2
        C2_minus = model.C2 - epsilon * delta_C2

        loss_plus = compute_loss(model.C1, C2_plus)
        loss_minus = compute_loss(model.C1, C2_minus)
        grad_C2_est = (loss_plus - loss_minus) / (2 * epsilon) * delta_C2

        # Gradient clipping for stability
        grad_C1_est = torch.clamp(grad_C1_est, -1.0, 1.0)
        grad_C2_est = torch.clamp(grad_C2_est, -1.0, 1.0)

        # Update correction matrices
        model.C1 -= lr * grad_C1_est
        model.C2 -= lr * grad_C2_est

# -----------------------------------------------------
# 4. Training loop
# -----------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

model = SPSA_CorrectedMLP(input_dim=1, hidden_dim=32, output_dim=1).to(device)

lr = 0.1
max_epochs = 5000

for epoch in range(max_epochs + 1):
    x_batch, y_batch = generate_data(128)
    x_batch, y_batch = x_batch.to(device), y_batch.to(device)

    spsa_update(model, x_batch, y_batch, lr)

    if epoch % 100 == 0:
        with torch.no_grad():
            y_pred = model(x_batch)
            loss = nn.MSELoss()(y_pred, y_batch).item()
            print(f"Epoch {epoch:4d} | Loss: {loss:.5f}")

# -----------------------------------------------------
# 5. Sanity plot
# -----------------------------------------------------
model.eval()
with torch.no_grad():
    xs = torch.linspace(-np.pi, np.pi, 400).unsqueeze(1).to(device)
    ys_pred = model(xs).cpu()
    ys_true = target_func(xs.cpu())

plt.figure(figsize=(6, 3))
plt.plot(xs.cpu(), ys_pred, label='SPSA-learned')
plt.plot(xs.cpu(), ys_true, '--', label=f'True {target_func.__name__}(x)')
plt.legend()
plt.title(f'SPSA Correction (Full C) after {max_epochs} epochs')
plt.show()
