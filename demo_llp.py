import numpy as np

# --- 1. Toy dataset ---
n_samples = 500
input_dim = 50
hidden_dims = [50]*10  # 10 layers with 50 units each
output_dim = 1

X = np.random.randn(n_samples, input_dim)
y = np.sin(X.sum(axis=1, keepdims=True))

# --- 2. Activation ---
def tanh(x):
    return np.tanh(x)

# --- 3. Generate hidden targets from [H_input, y] only ---
def generate_H(H_input, y, h_dim):
    XY = np.concatenate([H_input, y], axis=1)
    n_cols = XY.shape[1]

    # Orthogonal projection
    R_proj = np.random.randn(n_cols, h_dim)
    Q, _ = np.linalg.qr(R_proj)
    H = XY @ Q
    return H

# --- 4. Layerwise training ---
def train_layerwise(X, y, hidden_dims):
    layers = []
    H_input = X.copy()

    for h_dim in hidden_dims:
        H_target = generate_H(H_input, y, h_dim)

        # Center X and H
        X_mean = H_input.mean(axis=0, keepdims=True)
        H_mean = H_target.mean(axis=0, keepdims=True)
        X_centered = H_input - X_mean
        H_centered = H_target - H_mean

        # Solve W using pseudoinverse
        W = H_centered.T @ np.linalg.pinv(X_centered.T)
        b = H_mean.flatten() - W @ X_mean.flatten()

        # Compute plain residual
        pre_activation = H_input @ W.T + b
        E = H_target - tanh(pre_activation)

        # Forward pass
        H_input = tanh(pre_activation + E)

        layers.append((W, b, E))

    # Output layer (linear)
    X_mean = H_input.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)
    X_centered = H_input - X_mean
    y_centered = y - y_mean

    W_out = y_centered.T @ np.linalg.pinv(X_centered.T)
    b_out = y_mean.flatten() - W_out @ X_mean.flatten()

    layers.append((W_out, b_out))
    return layers

# --- 5. Forward pass ---
def forward_residual(X, layers):
    H = X.copy()
    for layer in layers[:-1]:
        W, b, E = layer
        pre_activation = H @ W.T + b
        H = tanh(pre_activation + E)
    W, b = layers[-1]
    y_pred = H @ W.T + b
    return y_pred

# --- 6. Train and evaluate ---
layers = train_layerwise(X, y, hidden_dims)
y_pred = forward_residual(X, layers)

print("True y[:5]:\n", y[:5])
print("\nPred y_pred[:5]:\n", y_pred[:5])
print("\nMSE:", np.mean((y - y_pred)**2))
