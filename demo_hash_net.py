import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def generate_data(n=512):
    x = torch.linspace(-np.pi, np.pi, n, device=device).unsqueeze(1)
    y = torch.sin(x)
    return x, y

X_train, Y_train = generate_data()

class LayerGridMLP:
    # 8 fixed values per layer
    K = 8
    grid = torch.linspace(-1.0, 1.0, K).to(device)

    def __init__(self):
        # one 3-bit index per weight
        self.h1 = np.random.randint(0, self.K, (32, 1), dtype=np.uint8)
        self.h2 = np.random.randint(0, self.K, (32, 32), dtype=np.uint8)
        self.h3 = np.random.randint(0, self.K, (1, 32), dtype=np.uint8)

    def _weight(self, h):
        idx = torch.from_numpy(h.astype(np.int64)).to(device)  # int64 indices
        return self.grid[idx]

    def __call__(self, x):
        W1 = self._weight(self.h1).view(32, 1)
        W2 = self._weight(self.h2).view(32, 32)
        W3 = self._weight(self.h3).view(1, 32)
        x = torch.sigmoid(F.linear(x, W1))
        x = torch.sigmoid(F.linear(x, W2))
        return F.linear(x, W3)

model = LayerGridMLP()
n_iter = 5000
losses = []

for step in range(n_iter + 1):
    for _ in range(3):
        choice = np.random.choice(['h1', 'h2', 'h3'])
        h = getattr(model, choice)
        i, j = np.random.randint(0, h.shape[0]), np.random.randint(0, h.shape[1])

        best_idx = h[i, j]
        best_loss = float('inf')
        for c in range(model.K):
            h[i, j] = c
            with torch.no_grad():
                loss = F.mse_loss(model(X_train), Y_train).item()
            if loss < best_loss:
                best_loss, best_idx = loss, c
        h[i, j] = best_idx

    if step % 100 == 0:
        with torch.no_grad():
            loss = F.mse_loss(model(X_train), Y_train).item()
        losses.append(loss)
        print(f'Iter {step:4d} | Loss {loss:.5f}')

with torch.no_grad():
    pred = model(X_train)

plt.figure(figsize=(8,4))
plt.plot(X_train.cpu(), Y_train.cpu(), label='Target')
plt.plot(X_train.cpu(), pred.cpu(), label='Predicted')
plt.legend(); plt.grid(); plt.show()

plt.figure(); plt.plot(losses); plt.title('Loss'); plt.grid(); plt.show()
