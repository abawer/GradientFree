import numpy as np

class Bucket:
    """Leaf or parent bucket with lazy linear regression"""
    def __init__(self, n_features, projections=None):
        self.n_features = n_features
        self.projections = [] if projections is None else list(projections)
        self.is_leaf = True
        self.samples_X = []
        self.samples_Y = []
        self.children = {}  # hash_code -> Bucket
        self.W = None
        self.b = None

    # ---------------- Add sample ----------------
    def add_sample(self, x, y):
        self.samples_X.append(x)
        self.samples_Y.append(y)

    # ---------------- Promote leaf to parent ----------------
    def promote(self):
        if not self.is_leaf:
            return  # already a parent
        new_proj = np.random.randn(self.n_features)
        self.projections.append(new_proj)
        self.is_leaf = False
        # redistribute samples into children
        for x, y in zip(self.samples_X, self.samples_Y):
            hash_code = int(np.floor(x @ new_proj))
            if hash_code not in self.children:
                self.children[hash_code] = Bucket(self.n_features, projections=self.projections.copy())
            self.children[hash_code].add_sample(x, y)
        # clear parent’s old leaf data
        self.samples_X = []
        self.samples_Y = []
        self.W = None
        self.b = None

    # ---------------- Fit linear model lazily ----------------
    def fit_linear(self):
        if not self.is_leaf or len(self.samples_X) == 0:
            return
        X = np.array(self.samples_X)
        Y = np.array(self.samples_Y)
        # append ones for bias
        X_aug = np.hstack([X, np.ones((X.shape[0],1))])
        Wb = np.linalg.pinv(X_aug) @ Y
        self.W = Wb[:-1]
        self.b = Wb[-1]

class AdaptiveHierarchicalHashRegressor:
    def __init__(self, max_samples_per_bucket=10):
        self.root = None
        self.x_min = None
        self.x_max = None
        self.samples_per_bucket = max_samples_per_bucket

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

        # Route samples to leaves
        for i in range(n_samples):
            x_i, y_i = X_norm[i], Y[i]
            bucket = self._route_to_leaf(x_i, create=True)
            bucket.add_sample(x_i, y_i)
            # optional promotion
            if len(bucket.samples_X) > self.samples_per_bucket:  # example threshold
                #print("splitting")
                bucket.promote()

        # After all samples are routed, fit leaf models
        self._fit_all_leaves(self.root)

    def _fit_all_leaves(self, bucket):
        if bucket.is_leaf:
            bucket.fit_linear()
        else:
            for child in bucket.children.values():
                self._fit_all_leaves(child)

    # ---------------- Routing ----------------
    def _route_to_leaf(self, x, create=False):
        bucket = self.root
        while not bucket.is_leaf:
            proj_val = float(x @ bucket.projections[-1])
            hash_code = int(np.floor(proj_val))
            if hash_code not in bucket.children:
                if create:
                    bucket.children[hash_code] = Bucket(bucket.n_features, projections=bucket.projections.copy())
                else:
                    # fallback: closest sibling
                    existing_hashes = np.array(list(bucket.children.keys()))
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
            if bucket.W is not None:
                Y_pred.append(float(x @ bucket.W + bucket.b))
            else:
                print("no pred!")
                Y_pred.append(0.0)
        return np.array(Y_pred).reshape(-1,1)

    def mse(self, X, Y):
        Y_pred = self.predict(X)
        return np.mean((Y - Y_pred)**2)

# ---------------- Demo ----------------
if __name__ == "__main__":
    ##np.random.seed(0)
    n_samples = 5000
    X = np.random.uniform(-2, 2, size=(n_samples, 2))
    Y = (np.sin(X[:,0]) + X[:,1]**2).reshape(-1,1) + 0.05*np.random.randn(n_samples,1)

    idx = np.random.permutation(n_samples)
    tsize = int(0.8 * n_samples)
    X_train, Y_train = X[idx[:tsize]], Y[idx[:tsize]]
    X_test, Y_test = X[idx[tsize:]], Y[idx[tsize:]]

    model = AdaptiveHierarchicalHashRegressor(max_samples_per_bucket=50)
    model.fit(X_train, Y_train)
    print("Train MSE:", model.mse(X_train, Y_train))
    print("Test  MSE:", model.mse(X_test, Y_test))
