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
    K = 8
    grid = torch.linspace(-2.0, 2.0, K).to(device)

    def __init__(self):
        # one 3-bit index per weight (including biases)
        self.h1 = np.random.randint(0, self.K, (32, 2), dtype=np.uint8)  # 32 weights + 1 bias
        self.h2 = np.random.randint(0, self.K, (32, 33), dtype=np.uint8) # 32 weights + 1 bias
        self.h3 = np.random.randint(0, self.K, (1, 33), dtype=np.uint8)  # 1 weight + 1 bias

    def _weight(self, h):
        idx = torch.from_numpy(h.astype(np.int64)).to(device)  # int64 indices
        return self.grid[idx]

    def __call__(self, x):
        W1 = self._weight(self.h1[:, :-1]).view(32, 1)  # exclude the last column (bias)
        b1 = self._weight(self.h1[:, -1]).view(32)      # last column is the bias

        W2 = self._weight(self.h2[:, :-1]).view(32, 32) # exclude the last column (bias)
        b2 = self._weight(self.h2[:, -1]).view(32)      # last column is the bias

        W3 = self._weight(self.h3[:, :-1]).view(1, 32)  # exclude the last column (bias)
        b3 = self._weight(self.h3[:, -1]).view(1)       # last column is the bias

        x = torch.sigmoid(F.linear(x, W1, b1))
        x = torch.sigmoid(F.linear(x, W2, b2))
        return F.linear(x, W3, b3)

model = LayerGridMLP()
n_iter = 5000
losses = []

attempts = 3
futile = 0

for step in range(n_iter + 1):    
    for _ in range(attempts):
        choice = np.random.choice(['h1', 'h2', 'h3'])
        h = getattr(model, choice)
        i = np.random.randint(0, h.shape[0])
        j = np.random.randint(0, h.shape[1])

        best_idx = h[i, j]
        best_loss = float('inf')
        for c in range(model.K):
            h[i, j] = c
            with torch.no_grad():
                loss = F.mse_loss(model(X_train), Y_train).item()
            if loss < best_loss:
                best_loss, best_idx = loss, c
        if h[i, j] != best_idx: 
            h[i, j] = best_idx
        else:
            futile += 1

    if step % 100 == 0:
        with torch.no_grad():
            loss = F.mse_loss(model(X_train), Y_train).item()
        losses.append(loss)
        print(f'Iter {step:4d} | Loss {loss:.5f} | Futile attempts {futile}/{100*attempts} ({100*futile/(100*attempts):.3f}%)')
        futile = 0

with torch.no_grad():
    pred = model(X_train)

plt.figure(figsize=(8,4))
plt.plot(X_train.cpu(), Y_train.cpu(), label='Target')
plt.plot(X_train.cpu(), pred.cpu(), label='Predicted')
plt.legend(); plt.grid(); plt.show()

plt.figure(); plt.plot(losses); plt.title('Loss'); plt.grid(); plt.show()
