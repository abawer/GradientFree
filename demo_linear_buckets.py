"""
Adaptive-Hierarchical-Hash Regressor – demo with leaf-region visualisation
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ---------------- Bucket ----------------
class Bucket:
    PROMOTE_EPS = 0.01
    MAX_DEPTH   = 40
    LAMBDA_REG  = 1e-3

    def __init__(self, n_features, projections=None, depth=0):
        if n_features <= 0:
            raise ValueError("n_features must be positive")
        self.n_features  = n_features
        self.projections = [] if projections is None else list(projections)
        self.depth       = depth
        self.is_leaf     = True
        self.samples_X   = []
        self.samples_Y   = []
        self.children    = {}
        self.W           = None
        self.b           = None

    # ------------- data ingestion -------------
    def add_sample(self, x, y):
        x = np.asarray(x, dtype=float)
        if x.shape != (self.n_features,):
            raise ValueError("x must be 1-D array of length n_features")
        y = float(y)
        self.samples_X.append(x)
        self.samples_Y.append(y)

    # ------------- ridge regression -------------
    def fit_linear(self):
        X = np.array(self.samples_X)
        Y = np.array(self.samples_Y)
        n = X.shape[0]
        if n == 0:
            raise RuntimeError("Cannot fit linear model: bucket has zero samples")
        X_aug = np.hstack([X, np.ones((n, 1))])
        reg   = self.LAMBDA_REG * np.eye(X_aug.shape[1])
        Wb    = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ Y)
        self.W, self.b = Wb[:-1], float(Wb[-1])

    # ------------- local MSE -------------
    def _mse_on_self(self):
        if len(self.samples_X) < 2 or self.W is None:
            return np.inf
        X = np.array(self.samples_X)
        Y = np.array(self.samples_Y)
        return float(np.mean((Y - (X @ self.W + self.b)) ** 2))

    # ------------- post-train splitting -------------
    def split_until_good(self):
        self.fit_linear()
        if (self.depth >= self.MAX_DEPTH or
            len(self.samples_X) < 2 or
            self._mse_on_self() <= self.PROMOTE_EPS):
            return

        new_proj = np.random.randn(self.n_features)
        self.projections.append(new_proj)
        self.is_leaf = False

        X = np.array(self.samples_X)
        Y = np.array(self.samples_Y)
        hash_codes = np.floor(X @ new_proj).astype(int)

        for hc in np.unique(hash_codes):
            mask = hash_codes == hc
            if mask.sum() == 0:
                continue
            child = Bucket(self.n_features,
                           projections=self.projections.copy(),
                           depth=self.depth + 1)
            child.samples_X = X[mask].tolist()
            child.samples_Y = Y[mask].tolist()
            self.children[hc] = child

        self.samples_X, self.samples_Y = [], []
        for child in self.children.values():
            child.split_until_good()

# ---------------- Adaptive Hierarchical Hash Regressor ----------------
class AdaptiveHierarchicalHashRegressor:
    def __init__(self):
        self.root  = None
        self.x_min = None
        self.x_max = None

    def _normalize(self, X):
        if self.x_min is None or self.x_max is None:
            raise RuntimeError("Model not fitted")
        return (X - self.x_min) / (self.x_max - self.x_min + 1e-12) * 2 - 1

    def fit(self, X, Y):
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float).ravel()
        if X.ndim != 2:
            raise ValueError("X must be 2-D")
        if Y.shape[0] != X.shape[0]:
            raise ValueError("X and Y must have same number of samples")
        n_samples, n_features = X.shape

        self.x_min = X.min(axis=0)
        self.x_max = X.max(axis=0)
        X_norm = self._normalize(X)

        self.root = Bucket(n_features)
        for x, y in zip(X_norm, Y):
            self.root.add_sample(x, y)

        self.root.split_until_good()

    def _route_to_leaf(self, x):
        if self.root is None:
            raise RuntimeError("Root bucket is None")
        bucket = self.root
        while not bucket.is_leaf:
            proj_val = float(x @ bucket.projections[-1])
            hc = int(np.floor(proj_val))
            if hc not in bucket.children:
                hc = min(bucket.children.keys(), key=lambda k: abs(k - hc))
            bucket = bucket.children[hc]

        if bucket.W is None:
            raise RuntimeError("Leaf bucket lacks a fitted model")
        return bucket

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("predict expects 2-D array")
        X_norm = self._normalize(X)
        return np.array([(leaf.W @ x) + leaf.b
                         for x, leaf in ((xi, self._route_to_leaf(xi)) for xi in X_norm)
                        ]).reshape(-1, 1)

    def mse(self, X, Y):
        Y = np.asarray(Y, dtype=float).ravel()
        Y_pred = self.predict(X).ravel()
        return float(np.mean((Y - Y_pred) ** 2))
    
    # ----------------------------------------------------------
    # Count leaves
    # ----------------------------------------------------------
    def count_leaves(self, node=None):
        node = node or self.root
        """Return the number of leaves in the tree rooted at *node*."""
        if node.is_leaf:
            return 1
        return sum(self.count_leaves(child) for child in node.children.values())

# ----------------------------------------------------------
# Demo
# ----------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0)

    # 1. Generate data
    n_samples = 5000
    X = np.random.uniform(-2, 2, size=(n_samples, 2))
    Y = (np.sin(X[:, 0]) + X[:, 1] ** 2) + 0.05 * np.random.randn(n_samples)

    # 2. Train/test split
    idx = np.random.permutation(n_samples)
    tsize = int(0.8 * n_samples)
    X_train, Y_train = X[idx[:tsize]], Y[idx[:tsize]]
    X_test,  Y_test  = X[idx[tsize:]], Y[idx[tsize:]]

    # 3. Fit model
    model = AdaptiveHierarchicalHashRegressor()
    model.fit(X_train, Y_train)
    print("Leaf count:", model.count_leaves())
    # 4. Evaluate
    print("Train MSE:", model.mse(X_train, Y_train))
    print("Test  MSE:", model.mse(X_test,  Y_test))

    # ----------------------------------------------------------
    # 5. Visualisation
    # ----------------------------------------------------------

    # 5a. Build a fine grid covering the test data range
    x0_min, x0_max = X_test[:, 0].min(), X_test[:, 0].max()
    x1_min, x1_max = X_test[:, 1].min(), X_test[:, 1].max()
    u = np.linspace(x0_min, x0_max, 400)
    v = np.linspace(x1_min, x1_max, 400)
    U, V = np.meshgrid(u, v)
    grid = np.c_[U.ravel(), V.ravel()]

    # 5b. Map every grid point to its leaf
    leaf_ids = np.empty(len(grid), dtype=int)
    for i, pt in enumerate(grid):
        leaf = model._route_to_leaf(model._normalize(pt.reshape(1, -1))[0])
        leaf_ids[i] = id(leaf)
    leaf_ids = leaf_ids.reshape(U.shape)

    # 5c. Unique colour for every leaf
    unique_ids = np.unique(leaf_ids)
    id_to_colour = {uid: i % 20 for i, uid in enumerate(unique_ids)}
    colour_map = np.vectorize(id_to_colour.get)(leaf_ids)

    # 5d. 2-D heat-map of leaf regions
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax0 = ax[0]
    ax0.imshow(colour_map,
               extent=[x0_min, x0_max, x1_min, x1_max],
               origin='lower',
               cmap='tab20',
               aspect='auto')
    ax0.set_title("Leaf regions (coloured by leaf id)")
    ax0.set_xlabel('x0'); ax0.set_ylabel('x1')

    # 5e. Overlay test points coloured by the same rule
    test_leaf_ids = np.array([id(model._route_to_leaf(model._normalize(pt.reshape(1, -1))[0]))
                              for pt in X_test])
    test_colours = np.vectorize(id_to_colour.get)(test_leaf_ids)
    ax0.scatter(X_test[:, 0], X_test[:, 1], c=test_colours, s=8,
                edgecolors='k', linewidths=0.2)

    # 5f. Draw the splitting boundaries (optional)
    def draw_boundaries(node):
        if node.is_leaf:
            return
        w = node.projections[-1]
        # map back to original scale
        w_orig = w / (model.x_max - model.x_min + 1e-12) * 2
        a, b = w_orig[0], w_orig[1]
        thresholds = sorted(node.children.keys())
        xs = np.linspace(x0_min, x0_max, 400)
        for k in thresholds:
            if abs(b) > 1e-8:
                ys = (k - a * xs) / b
                ys = np.clip(ys, x1_min, x1_max)
                ax0.plot(xs, ys, 'k-', linewidth=0.5, alpha=0.4)
            elif abs(a) > 1e-8:
                xs_line = k / a
                if x0_min <= xs_line <= x0_max:
                    ax0.axvline(xs_line, color='k', linewidth=0.5, alpha=0.4)
        for child in node.children.values():
            draw_boundaries(child)

    draw_boundaries(model.root)

    # 5g. 3-D surface on the right
    ax2 = fig.add_subplot(122, projection='3d')
    Z_pred = model.predict(np.c_[U.ravel(), V.ravel()]).reshape(U.shape)
    ax2.plot_surface(U, V, Z_pred, cmap='viridis', alpha=0.7)
    ax2.scatter(X_test[:, 0], X_test[:, 1], Y_test, s=6, color='r', alpha=0.5)
    ax2.set_title("Predicted surface + true test points")
    ax2.set_xlabel('x0'); ax2.set_ylabel('x1'); ax2.set_zlabel('Y')

    plt.tight_layout()
    plt.show()
