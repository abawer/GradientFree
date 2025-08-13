import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    # denser grid around 0
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

    def __call__(self, x):
        W1 = self._weight(self.h1[:, :-1]).view(self.WIDTH, 1)
        b1 = self._weight(self.h1[:, -1]).view(self.WIDTH)

        W2 = self._weight(self.h2[:, :-1]).view(self.WIDTH, self.WIDTH)
        b2 = self._weight(self.h2[:, -1]).view(self.WIDTH)

        W3 = self._weight(self.h3[:, :-1]).view(1, self.WIDTH)
        b3 = self._weight(self.h3[:, -1]).view(1)

        x = torch.tanh(F.linear(x, W1, b1))
        x = torch.tanh(F.linear(x, W2, b2))
        return F.linear(x, W3, b3)

model = LayerGridMLP()

# ------------------------------------------------------------------
# Training: single-weight Gibbs sampling
# ------------------------------------------------------------------
losses = []
n_iter = 1000

for step in range(n_iter + 1):
    for layer in ['h1','h2','h3']:
        h = getattr(model, layer).copy()
        base_loss = F.mse_loss(model(X_train), Y_train).item()

        # pick one random index
        idx = tuple(np.unravel_index(np.random.randint(h.size), h.shape))
        old = h[idx]

        # brute-force best code at this index
        for code in range(LayerGridMLP.K):
            if code == old:
                continue
            h[idx] = code
            setattr(model, layer, h)
            new_loss = F.mse_loss(model(X_train), Y_train).item()
            if new_loss < base_loss:
                base_loss = new_loss
                old = h[idx]
            else:
                h[idx] = old  # revert to best so far

    # logging
    if step % 100 == 0:
        losses.append(base_loss)
        print(f'Iter {step:4d} | Loss {base_loss:.6f}')

# ------------------------------------------------------------------
# Visualisation
# ------------------------------------------------------------------
with torch.no_grad():
    pred = model(X_train)

plt.figure(figsize=(8, 4))
plt.plot(X_train.cpu(), Y_train.cpu(), label='Target')
plt.plot(X_train.cpu(), pred.cpu(), label='Predicted')
plt.legend(); plt.grid(); plt.show()

plt.figure()
plt.plot(losses)
plt.title('MSE vs. iteration'); plt.grid(); plt.show()
