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
# 2. Network with frozen noisy weights and Low-Rank correction matrices
# -----------------------------------------------------
class SPSA_LowRankMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, low_rank=16):
        super().__init__()
        # Fixed noisy weights (frozen)
        self.W1_noisy = nn.Parameter(torch.randn(hidden_dim, input_dim+1), requires_grad=False)
        self.W2_noisy = nn.Parameter(torch.randn(output_dim, hidden_dim+1), requires_grad=False)

        # Low-Rank factors U and V
        self.U1 = nn.Parameter(torch.eye(hidden_dim, low_rank))
        self.V1 = nn.Parameter(torch.eye(hidden_dim, low_rank))
        self.U2 = nn.Parameter(torch.eye(output_dim, low_rank))
        self.V2 = nn.Parameter(torch.eye(output_dim, low_rank))

    def forward(self, x):
        x1 = torch.cat([x, torch.ones(x.size(0), 1, device=x.device)], dim=1)

        # Low-Rank Correction Matrices
        C1 = self.U1 @ self.V1.T  # (hidden_dim x hidden_dim)
        W1_corr = C1 @ self.W1_noisy
        h = torch.sigmoid(x1 @ W1_corr.t())

        h1 = torch.cat([h, torch.ones(h.size(0), 1, device=h.device)], dim=1)

        C2 = self.U2 @ self.V2.T  # (output_dim x output_dim)
        W2_corr = C2 @ self.W2_noisy
        y_pred = h1 @ W2_corr.t()
        return y_pred

# -----------------------------------------------------
# 3. SPSA update for Low-Rank factors
# -----------------------------------------------------
def spsa_update(model, x, y_target, lr, epsilon=1e-4):
    criterion = nn.MSELoss()
    with torch.no_grad():
        # Helper to compute loss
        def compute_loss():
            y_pred = model(x)
            return criterion(y_pred, y_target).item()

        # SPSA perturbation for all factors simultaneously
        def spsa_step(param):
            delta = torch.randint(0, 2, param.shape, device=param.device, dtype=torch.float32) * 2 - 1  # ±1
            param_plus = param + epsilon * delta
            param_minus = param - epsilon * delta
            return delta, param_plus, param_minus

        # Perturb all U and V
        delta_U1, U1_plus, U1_minus = spsa_step(model.U1)
        delta_V1, V1_plus, V1_minus = spsa_step(model.V1)
        delta_U2, U2_plus, U2_minus = spsa_step(model.U2)
        delta_V2, V2_plus, V2_minus = spsa_step(model.V2)

        # Compute losses for perturbed parameters
        # Perturb all simultaneously (one SPSA step)
        model.U1.copy_(U1_plus)
        model.V1.copy_(V1_plus)
        model.U2.copy_(U2_plus)
        model.V2.copy_(V2_plus)
        loss_plus = compute_loss()

        model.U1.copy_(U1_minus)
        model.V1.copy_(V1_minus)
        model.U2.copy_(U2_minus)
        model.V2.copy_(V2_minus)
        loss_minus = compute_loss()

        # Restore original params
        model.U1.copy_((U1_plus + U1_minus) / 2)
        model.V1.copy_((V1_plus + V1_minus) / 2)
        model.U2.copy_((U2_plus + U2_minus) / 2)
        model.V2.copy_((V2_plus + V2_minus) / 2)

        # Estimate gradients and update
        grad_U1 = (loss_plus - loss_minus) / (2 * epsilon) * delta_U1
        grad_V1 = (loss_plus - loss_minus) / (2 * epsilon) * delta_V1
        grad_U2 = (loss_plus - loss_minus) / (2 * epsilon) * delta_U2
        grad_V2 = (loss_plus - loss_minus) / (2 * epsilon) * delta_V2

        # Gradient clipping for stability
        grad_U1 = torch.clamp(grad_U1, -1.0, 1.0)
        grad_V1 = torch.clamp(grad_V1, -1.0, 1.0)
        grad_U2 = torch.clamp(grad_U2, -1.0, 1.0)
        grad_V2 = torch.clamp(grad_V2, -1.0, 1.0)

        # Update factors
        model.U1 -= lr * grad_U1
        model.V1 -= lr * grad_V1
        model.U2 -= lr * grad_U2
        model.V2 -= lr * grad_V2

# -----------------------------------------------------
# 4. Training loop
# -----------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using device:', device)

model = SPSA_LowRankMLP(input_dim=1, hidden_dim=32, output_dim=1, low_rank=4).to(device)

lr = 0.01
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
plt.plot(xs.cpu(), ys_pred, label='SPSA Low-Rank Learned')
plt.plot(xs.cpu(), ys_true, '--', label=f'True {target_func.__name__}(x)')
plt.legend()
plt.title(f'SPSA Low-Rank Correction after {max_epochs} epochs')
plt.show()
