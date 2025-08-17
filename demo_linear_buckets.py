import numpy as np
from collections import defaultdict

class HashLinearRegressorSimple:
    def __init__(self, n_hashes=8, bin_width=1.2):
        self.n_hashes = n_hashes
        self.bin_width = bin_width
        self.projections = None
        self.buckets = {}        # hash tuple -> (W, b)
        self.bucket_codes = None
        self.bucket_counts = {}  # hash tuple -> number of samples

    def _hash(self, X):
        codes = np.floor(X @ self.projections.T / self.bin_width).astype(int)
        return [tuple(row) for row in codes]

    def _fit_linear(self, X, Y):
        W = np.linalg.pinv(X) @ Y
        b = np.zeros(Y.shape[1]) if Y.ndim > 1 else 0
        return W, b

    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.projections = np.random.randn(self.n_hashes, n_features)

        hash_codes = self._hash(X)
        buckets_indices = defaultdict(list)
        for i, code in enumerate(hash_codes):
            buckets_indices[code].append(i)

        for code, indices in buckets_indices.items():
            X_bucket = X[indices]
            Y_bucket = Y[indices]
            W, b = self._fit_linear(X_bucket, Y_bucket)
            self.buckets[code] = (W, b)
            self.bucket_counts[code] = len(indices)

        self.bucket_codes = np.array(list(self.buckets.keys()))

    def _nearest_bucket(self, code):
        code_array = np.array(code)
        dists = np.sum(np.abs(self.bucket_codes - code_array), axis=1)
        nearest_idx = np.argmin(dists)
        nearest_code = tuple(self.bucket_codes[nearest_idx])
        return self.buckets[nearest_code]

    def predict(self, X):
        hash_codes = self._hash(X)
        Y_pred = []
        for i, code in enumerate(hash_codes):
            W, b = self.buckets.get(code, self._nearest_bucket(code))
            y = X[i] @ W + b
            Y_pred.append(y)
        return np.vstack(Y_pred)

    def mse(self, X, Y):
        Y_pred = self.predict(X)
        return np.mean((Y - Y_pred) ** 2)

    # ---------------- Printing helpers ----------------
    def print_bucket_counts(self):
        print("Bucket counts (hash_code -> number of samples):")
        for code, count in self.bucket_counts.items():
            print(code, ":", count)

    def print_bucket_histogram(self, bin_size=2):
        counts = list(self.bucket_counts.values())
        max_count = max(counts)
        min_count = min(counts)

        print("\nBucket size histogram:")
        for size in range(min_count, max_count + 1, bin_size):
            num_buckets = sum(1 for c in counts if size <= c < size + bin_size)
            if num_buckets > 0:
                print(f"Samples {size:3d}-{size+bin_size-1:3d}: {'*' * num_buckets} ({num_buckets} buckets)")


# ---------------- Demo ----------------
if __name__ == "__main__":
    np.random.seed(0)

    # Generate 2D input
    n_samples = 5000
    tsize = int(0.8 * n_samples)
    X = np.random.uniform(-2, 2, size=(n_samples, 2))

    # Non-linear target: sin(x0) + x1^2
    Y = (np.sin(X[:,0]) + X[:,1]**2).reshape(-1,1) + 0.05*np.random.randn(n_samples,1)

    # Split train/test
    idx = np.random.permutation(n_samples)
    train_idx, test_idx = idx[:tsize], idx[tsize:]
    X_train, Y_train = X[train_idx], Y[train_idx]
    X_test, Y_test = X[test_idx], Y[test_idx]

    # Initialize and fit model
    model = HashLinearRegressorSimple(n_hashes=8, bin_width=1.2)
    model.fit(X_train, Y_train)

    # Predict and evaluate
    print("Train MSE:", model.mse(X_train, Y_train))
    print("Test  MSE:", model.mse(X_test, Y_test))

    # Print bucket info
    model.print_bucket_counts()
    model.print_bucket_histogram(bin_size=2)
