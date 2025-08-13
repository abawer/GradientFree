import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
device = 'cpu'  # you said CPU only

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
    WIDTH = 32
    K = 10
    half = torch.logspace(-2, 0, K // 2, base=10)
    grid = torch.cat([-half.flip(0), torch.zeros(1), half]) * 2.0
    grid = grid.to(device)

    def __init__(self):
        self.h1 = np.random.randint(0, self.K, (self.WIDTH, 2), dtype=np.uint8)
        self.h2 = np.random.randint(0, self.K, (self.WIDTH, self.WIDTH + 1), dtype=np.uint8)
        self.h3 = np.random.randint(0, self.K, (1, self.WIDTH + 1), dtype=np.uint8)

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

    # Forward from precomputed a1
    def forward_from_a1(self, a1):
        W2 = self._weight(self.h2[:, :-1]).view(self.WIDTH, self.WIDTH)
        b2 = self._weight(self.h2[:, -1])
        a2 = torch.tanh(F.linear(a1, W2, b2))

        W3 = self._weight(self.h3[:, :-1]).view(1, self.WIDTH)
        b3 = self._weight(self.h3[:, -1])
        return F.linear(a2, W3, b3)

    # Forward from precomputed a2
    def forward_from_a2(self, a2):
        W3 = self._weight(self.h3[:, :-1]).view(1, self.WIDTH)
        b3 = self._weight(self.h3[:, -1])
        return F.linear(a2, W3, b3)

    # Try all possible codes at given index in a layer
    def forward_with_h_change(self, layer_name, idx, X_train):
        codes = torch.arange(self.K, device=device)

        # Keep a copy
        h_original = getattr(self, layer_name).copy()

        # Expand into K versions
        h_variants = np.repeat(h_original[np.newaxis, ...], self.K, axis=0)
        h_variants[:, idx[0], idx[1]] = np.arange(self.K, dtype=np.uint8)

        outputs = []

        if layer_name == 'h1':
            for h in h_variants:
                self.h1 = h
                outputs.append(self.forward(X_train))

        elif layer_name == 'h2':
            # Cache a1 once
            a1 = torch.tanh(F.linear(
                X_train,
                self._weight(self.h1[:, :-1]).view(self.WIDTH, 1),
                self._weight(self.h1[:, -1])
            ))
            for h in h_variants:
                self.h2 = h
                outputs.append(self.forward_from_a1(a1))

        elif layer_name == 'h3':
            # Cache a1 and a2 once
            a1 = torch.tanh(F.linear(
                X_train,
                self._weight(self.h1[:, :-1]).view(self.WIDTH, 1),
                self._weight(self.h1[:, -1])
            ))
            a2 = torch.tanh(F.linear(
                a1,
                self._weight(self.h2[:, :-1]).view(self.WIDTH, self.WIDTH),
                self._weight(self.h2[:, -1])
            ))
            for h in h_variants:
                self.h3 = h
                outputs.append(self.forward_from_a2(a2))

        else:
            raise ValueError(f"Unknown layer: {layer_name}")

        # Restore original h
        setattr(self, layer_name, h_original)

        return torch.stack(outputs, dim=0)  # shape: [K, N, 1]


# ------------------------------------------------------------------
# Training loop
# ------------------------------------------------------------------
model = LayerGridMLP()
losses = []
n_iter = 1000

for step in range(n_iter + 1):
    for layer in ['h1', 'h2', 'h3']:
        h = getattr(model, layer)
        idx = tuple(np.unravel_index(np.random.randint(h.size), h.shape))

        out_all = model.forward_with_h_change(layer, idx, X_train)
        mse_all = ((out_all - Y_train.unsqueeze(0)) ** 2).mean(dim=(1, 2))
        best_code = torch.argmin(mse_all).item()

        getattr(model, layer)[idx] = best_code

    if step % 100 == 0:        
        final_pred = out_all[best_code]  # from last layer update
        loss = F.mse_loss(final_pred, Y_train).item()
        losses.append(loss)
        print(f"Iter {step:4d} | Loss {loss:.6f}")

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
