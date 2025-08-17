import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------- Bucket ----------------
class Bucket:
    """Leaf or parent bucket with lazy linear regression + MSE-driven promotion"""
    PROMOTE_M      = 32       # check every m new samples
    PROMOTE_EPS    = 0.002     # promote if training MSE > eps
    LAMBDA_REG     = 1e-3     # ridge regulariser

    def __init__(self, n_features, projections=None):
        self.n_features = n_features
        self.projections = [] if projections is None else list(projections)
        self.is_leaf = True
        self.samples_X = []
        self.samples_Y = []
        self.children = {}  # hash_code -> Bucket
        self.W = None
        self.b = None
        self._since_last_check = 0

    # ---------------- Add sample ----------------
    def add_sample(self, x, y):
        self.samples_X.append(x)
        self.samples_Y.append(y)
        self._since_last_check += 1

    # ---------------- Promote leaf to parent ----------------
    def promote(self):
        if not self.is_leaf:
            raise RuntimeError("Cannot promote: bucket is already a parent!")
        new_proj = np.random.randn(self.n_features)
        self.projections.append(new_proj)
        self.is_leaf = False
        # redistribute samples into children
        for x, y in zip(self.samples_X, self.samples_Y):
            hash_code = int(np.floor(x @ new_proj))
            if hash_code not in self.children:
                self.children[hash_code] = Bucket(self.n_features,
                                                  projections=self.projections.copy())
            self.children[hash_code].add_sample(x, y)
        # clear parent’s old leaf data
        self.samples_X = []
        self.samples_Y = []
        self.W = None
        self.b = None

    # ---------------- Fit linear model lazily with ridge ----------------
    def fit_linear(self):
        if not self.is_leaf:
            raise RuntimeError("fit_linear called on non-leaf bucket!")
        if len(self.samples_X) == 0:
            raise ValueError("Cannot fit linear model: leaf has zero samples!")
        X = np.array(self.samples_X)
        Y = np.array(self.samples_Y)
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        Wb = np.linalg.inv(X_aug.T @ X_aug +
                           self.LAMBDA_REG * np.eye(X_aug.shape[1])) @ X_aug.T @ Y
        self.W = Wb[:-1]
        self.b = Wb[-1]

    # ---------------- MSE on node’s own samples ----------------
    def _mse_on_self(self):
        """Return training-set MSE using current (W,b)."""
        if len(self.samples_X) < 2 or self.W is None:
            return np.inf
        X = np.array(self.samples_X)
        Y = np.array(self.samples_Y)
        pred = X @ self.W + self.b
        return float(np.mean((Y - pred) ** 2))

    # ---------------- Decide promotion ----------------
    def maybe_promote(self):
        """Return True if promotion triggered."""
        if not self.is_leaf or self._since_last_check < self.PROMOTE_M:
            return False
        self._since_last_check = 0
        self.fit_linear()
        if self._mse_on_self() > self.PROMOTE_EPS:
            self.promote()
            return True
        return False

# ---------------- Adaptive Hierarchical Hash Regressor ----------------
class AdaptiveHierarchicalHashRegressor:
    def __init__(self):
        self.root = None
        self.x_min = None
        self.x_max = None

    def _normalize(self, X):
        return (X - self.x_min) / (self.x_max - self.x_min + 1e-12) * 2 - 1

    # ---------------- Training ----------------
    def fit(self, X, Y):
        n_samples, n_features = X.shape
        self.x_min = X.min(axis=0)
        self.x_max = X.max(axis=0)
        X_norm = self._normalize(X)

        if self.root is None:
            self.root = Bucket(n_features)

        for i in range(n_samples):
            x_i, y_i = X_norm[i], Y[i]
            bucket = self._route_to_leaf(x_i, create=True)
            bucket.add_sample(x_i, y_i)
            bucket.maybe_promote()

        self._fit_all_leaves_and_clear_samples(self.root)

    def _fit_all_leaves_and_clear_samples(self, bucket):
        if bucket.is_leaf:
            bucket.fit_linear()
            bucket.samples_X = []
            bucket.samples_Y = []
        else:
            for child in bucket.children.values():
                self._fit_all_leaves_and_clear_samples(child)
    
    # ---------------- Routing ----------------
    def _route_to_leaf(self, x, create=False):
        bucket = self.root
        if bucket is None:
            raise RuntimeError("Root bucket is None!")
        while not bucket.is_leaf:
            proj_val = (x @ bucket.projections[-1]).item()
            hash_code = int(np.floor(proj_val))
            if hash_code not in bucket.children:
                if create:
                    bucket.children[hash_code] = Bucket(bucket.n_features,
                                                        projections=bucket.projections.copy())
                else:
                    existing_hashes = np.array(list(bucket.children.keys()))
                    if len(existing_hashes) == 0:
                        raise RuntimeError("No children to fallback to!")
                    nearest_idx = np.argmin(np.abs(existing_hashes - proj_val))
                    hash_code = existing_hashes[nearest_idx]
            bucket = bucket.children[hash_code]
        return bucket

    # ---------------- Prediction ----------------
    def predict(self, X):
        X_norm = self._normalize(X)
        Y_pred = []
        for x in X_norm:
            bucket = self._route_to_leaf(x, create=False)
            if bucket.W is None:
                raise RuntimeError("Predicting from leaf without model!")
            Y_pred.append((x @ bucket.W + bucket.b).item())
        return np.array(Y_pred).reshape(-1, 1)

    def mse(self, X, Y):
        Y_pred = self.predict(X)
        return np.mean((Y - Y_pred) ** 2)

# ---------------- Demo ----------------
if __name__ == "__main__":
    np.random.seed(0)

    # 1. Generate data
    n_samples = 5000
    X = np.random.uniform(-2, 2, size=(n_samples, 2))
    Y = (np.sin(X[:, 0]) + X[:, 1] ** 2).reshape(-1, 1) + \
        0.05 * np.random.randn(n_samples, 1)

    # 2. Train/test split
    idx = np.random.permutation(n_samples)
    tsize = int(0.8 * n_samples)
    X_train, Y_train = X[idx[:tsize]], Y[idx[:tsize]]
    X_test,  Y_test  = X[idx[tsize:]], Y[idx[tsize:]]

    # 3. Fit model
    model = AdaptiveHierarchicalHashRegressor()
    model.fit(X_train, Y_train)

    # 4. Evaluate
    print("Train MSE:", model.mse(X_train, Y_train))
    print("Test  MSE:", model.mse(X_test, Y_test))

    # 5. Plot: true test points + predicted surface on the same x/y axes
    fig = plt.figure(figsize=(14, 5))

    # ---- 5a. True test values (scatter) ----
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(X_test[:, 0], X_test[:, 1], Y_test,
                s=8, c='k', alpha=0.6)
    ax1.set_title("True test values")
    ax1.set_xlabel('x0'); ax1.set_ylabel('x1'); ax1.set_zlabel('Y')

    # ---- 5b. Predicted surface (mesh) ----
    ax2 = fig.add_subplot(122, projection='3d')
    x0_min, x0_max = X_test[:, 0].min(), X_test[:, 0].max()
    x1_min, x1_max = X_test[:, 1].min(), X_test[:, 1].max()
    u = np.linspace(x0_min, x0_max, 100)
    v = np.linspace(x1_min, x1_max, 100)
    U, V = np.meshgrid(u, v)
    grid = np.c_[U.ravel(), V.ravel()]
    Z_pred = model.predict(grid).reshape(U.shape)
    ax2.plot_surface(U, V, Z_pred, cmap='viridis', alpha=0.7)
    ax2.scatter(X_test[:, 0], X_test[:, 1], Y_test,
                s=6, color='r', alpha=0.5)
    ax2.set_title("Predicted surface + true test points")
    ax2.set_xlabel('x0'); ax2.set_ylabel('x1'); ax2.set_zlabel('Y')
    plt.tight_layout()
    plt.show()
