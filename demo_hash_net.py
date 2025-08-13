import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import random

torch.manual_seed(42)
device = 'cpu'

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
def generate_data(n=512):
    x = torch.linspace(-np.pi, np.pi, n, device=device).unsqueeze(1)
    y = torch.sin(x)
    return x, y

X_train, Y_train = generate_data()

# ------------------------------------------------------------------
# Model definition
# ------------------------------------------------------------------
class LayerGridMLP:
    WIDTH = 16
    K = 2000
    half = torch.logspace(-2, 0, K // 2, base=10)
    grid = torch.cat([-half.flip(0), torch.zeros(1), half]) * 2.0
    grid = grid.to(device)

    def __init__(self):
        self.h1 = np.random.randint(0, self.K, (self.WIDTH, 2), dtype=np.uint32)
        self.h2 = np.random.randint(0, self.K, (self.WIDTH, self.WIDTH + 1), dtype=np.uint32)
        self.h3 = np.random.randint(0, self.K, (1, self.WIDTH + 1), dtype=np.uint32)

        # Cached activations after committed changes
        self.a1_cache = None
        self.a2_cache = None

    def _weight(self, h):
        idx = torch.from_numpy(h.astype(np.int64)).to(device)
        return self.grid[idx]

    # Standard forward
    def forward(self, x):
        W1 = self._weight(self.h1[:, :-1]).view(self.WIDTH, 1)
        b1 = self._weight(self.h1[:, -1])
        x = torch.tanh(F.linear(x, W1, b1))

        W2 = self._weight(self.h2[:, :-1]).view(self.WIDTH, self.WIDTH)
        b2 = self._weight(self.h2[:, -1])
        x = torch.tanh(F.linear(x, W2, b2))

        W3 = self._weight(self.h3[:, :-1]).view(1, self.WIDTH)
        b3 = self._weight(self.h3[:, -1])
        return F.linear(x, W3, b3)

    def forward_with_single_code(self, layer_name, idx, code, X_train):
        """
        Compute forward for a single candidate code.
        Only recompute affected activations; reuse caches when possible.
        """
        h_original = getattr(self, layer_name).copy()
        h = h_original.copy()
        h[idx] = code
        setattr(self, layer_name, h)

        # --- Compute a1 ---
        if layer_name == 'h1' or self.a1_cache is None:
            W1 = self._weight(self.h1[:, :-1]).view(self.WIDTH, 1)
            b1 = self._weight(self.h1[:, -1])
            a1 = torch.tanh(F.linear(X_train, W1, b1))
        else:
            a1 = self.a1_cache  # reuse cache

        # --- Compute a2 ---
        if layer_name in ['h1', 'h2'] or self.a2_cache is None:
            W2 = self._weight(self.h2[:, :-1]).view(self.WIDTH, self.WIDTH)
            b2 = self._weight(self.h2[:, -1])
            a2 = torch.tanh(F.linear(a1, W2, b2))
        else:
            a2 = self.a2_cache  # reuse cache

        # --- Compute output ---
        W3 = self._weight(self.h3[:, :-1]).view(1, self.WIDTH)
        b3 = self._weight(self.h3[:, -1])
        out = F.linear(a2, W3, b3)

        setattr(self, layer_name, h_original)
        return out

    # Update caches after committing a new code
    def update_caches(self):
        W1 = self._weight(self.h1[:, :-1]).view(self.WIDTH, 1)
        b1 = self._weight(self.h1[:, -1])
        self.a1_cache = torch.tanh(F.linear(X_train, W1, b1))

        W2 = self._weight(self.h2[:, :-1]).view(self.WIDTH, self.WIDTH)
        b2 = self._weight(self.h2[:, -1])
        self.a2_cache = torch.tanh(F.linear(self.a1_cache, W2, b2))


# ------------------------------------------------------------------
# Binary search over the grid
# ------------------------------------------------------------------
def best_code_binary(model, layer_name, idx, X_train, Y_train):
    lo, hi = 0, model.K - 1
    best_code = lo
    best_loss = float("inf")

    while lo <= hi:
        mid = (lo + hi) // 2
        out_mid = model.forward_with_single_code(layer_name, idx, mid, X_train)
        loss_mid = F.mse_loss(out_mid, Y_train).item()

        if mid + 1 <= hi:
            out_next = model.forward_with_single_code(layer_name, idx, mid + 1, X_train)
            loss_next = F.mse_loss(out_next, Y_train).item()
        else:
            loss_next = float("inf")

        if loss_mid < best_loss:
            best_loss = loss_mid
            best_code = mid
        if loss_next < best_loss:
            best_loss = loss_next
            best_code = mid + 1

        if loss_next < loss_mid:
            lo = mid + 1
        else:
            hi = mid - 1

    return best_code, best_loss


# ------------------------------------------------------------------
# Training loop with monotonic loss + smart caching
# ------------------------------------------------------------------
model = LayerGridMLP()
losses = []
n_iter = 2000

for step in range(n_iter + 1):
    layer = random.choice(['h1', 'h2', 'h3'])
    h = getattr(model, layer)
    idx = tuple(np.unravel_index(np.random.randint(h.size), h.shape))

    old_code = h[idx]
    out_old = model.forward_with_single_code(layer, idx, old_code, X_train)
    loss_old = F.mse_loss(out_old, Y_train).item()

    best_code, _ = best_code_binary(model, layer, idx, X_train, Y_train)
    out_new = model.forward_with_single_code(layer, idx, best_code, X_train)
    loss_new = F.mse_loss(out_new, Y_train).item()

    if loss_new < loss_old:
        getattr(model, layer)[idx] = best_code
        model.update_caches()  # recompute caches only once
        final_loss = loss_new
    else:
        final_loss = loss_old

    if step % 100 == 0:
        losses.append(final_loss)
        print(f"Iter {step:4d} | Loss {final_loss:.6f}")


# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------
with torch.no_grad():
    pred = model.forward(X_train)

plt.figure(figsize=(8, 4))
plt.plot(X_train.cpu(), Y_train.cpu(), label='Target')
plt.plot(X_train.cpu(), pred.cpu(), label='Predicted')
plt.legend(); plt.grid(); plt.show()

plt.figure()
plt.plot(losses)
plt.title('MSE vs. iteration'); plt.grid(); plt.show()
