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
# 2. Network with frozen noisy weights and Low-Rank additive correction matrices
# -----------------------------------------------------
class SPSA_LowRankMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, low_rank=16):
        super().__init__()
        # Fixed noisy weights (frozen)
        self.W1_noisy = nn.Parameter(torch.randn(hidden_dim, input_dim + 1), requires_grad=False)
        self.W2_noisy = nn.Parameter(torch.randn(output_dim, hidden_dim + 1), requires_grad=False)

        # Low-Rank factors U and V for additive corrections
        self.U1 = nn.Parameter(torch.randn(hidden_dim, low_rank) * 0.01)
        self.V1 = nn.Parameter(torch.randn(low_rank, input_dim + 1) * 0.01)
        self.U2 = nn.Parameter(torch.randn(output_dim, low_rank) * 0.01)
        self.V2 = nn.Parameter(torch.randn(low_rank, hidden_dim + 1) * 0.01)

    def forward(self, x):
        x1 = torch.cat([x, torch.ones(x.size(0), 1, device=x.device)], dim=1)  # add bias term

        # Low-Rank Correction Matrices (additive)
        W1_corr = self.W1_noisy + (self.U1 @ self.V1)  # (hidden_dim x (input_dim+1))
        h = torch.sigmoid(x1 @ W1_corr.t())

        h1 = torch.cat([h, torch.ones(h.size(0), 1, device=h.device)], dim=1)  # add bias term

        W2_corr = self.W2_noisy + (self.U2 @ self.V2)  # (output_dim x (hidden_dim+1))
        y_pred = h1 @ W2_corr.t()
        return y_pred

# -----------------------------------------------------
# 3. SPSA update per layer (individual perturbations)
# -----------------------------------------------------
def spsa_update(model, x, y_target, lr, epsilon=1e-4):
    criterion = nn.MSELoss()
    with torch.no_grad():
        def compute_loss():
            y_pred = model(x)
            return criterion(y_pred, y_target).item()

        layers = ['U1', 'V1', 'U2', 'V2']
        grads = {}

        for layer_name in layers:
            param = getattr(model, layer_name)
            delta = torch.randint(0, 2, param.shape, device=param.device, dtype=torch.float32) * 2 - 1

            param_plus = param + epsilon * delta
            param_minus = param - epsilon * delta

            # Save original param
            param_orig = param.clone()

            # Perturb +epsilon
            param.copy_(param_plus)
            loss_plus = compute_loss()

            # Perturb -epsilon
            param.copy_(param_minus)
            loss_minus = compute_loss()

            # Restore original param
            param.copy_(param_orig)

            # Estimate gradient
            grad = (loss_plus - loss_minus) / (2 * epsilon) * delta
            grad = torch.clamp(grad, -1.0, 1.0)
            grads[layer_name] = grad

        # Update parameters
        for layer_name in layers:
            param = getattr(model, layer_name)
            param -= lr * grads[layer_name]

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
plt.title(f'SPSA Low-Rank Additive Correction after {max_epochs} epochs')
plt.show()
