"""
trees.py - Decision Tree, Random Forest and LDA written from scratch with NumPy.

Used by decision_tree/decision_tree.ipynb, random_forest/random_forest.ipynb and final/ensemble.ipynb.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np


# ---------------------------------------------------------------- impurity
# counts[..., c] = number of rows of class c (last axis = classes)

def gini(counts):
    """Gini impurity: 1 - sum(p_c^2)."""
    p = counts / np.maximum(counts.sum(axis=-1, keepdims=True), 1)
    return 1.0 - (p ** 2).sum(axis=-1)


def entropy(counts):
    """Entropy: -sum(p_c * log2(p_c))."""
    p = counts / np.maximum(counts.sum(axis=-1, keepdims=True), 1)
    log_p = np.log2(np.where(p > 0, p, 1.0))     # empty classes add 0
    return -(p * log_p).sum(axis=-1)


IMPURITY = {"gini": gini, "entropy": entropy}


# ---------------------------------------------------------------- thresholds
class Binner:
    """Candidate thresholds per feature, learned from the training rows only."""

    def __init__(self, n_bins=64):
        assert 2 <= n_bins <= 256, "bin numbers are stored as uint8"
        self.n_bins = n_bins

    def fit(self, X):
        q = np.linspace(0, 1, self.n_bins + 1)[1:-1]             # quantile levels
        thresholds = []
        for f in range(X.shape[1]):
            col = X[:, f]
            cand = np.quantile(col, q)
            nonzero = col[col != 0]
            if 0 < len(nonzero) < len(col):                       # mostly-zero feature: add non-zero quantiles
                cand = np.concatenate([cand, np.quantile(nonzero, q)])
            cand = np.unique(cand)
            cand = cand[cand < col.max()]                         # a threshold at the max splits nothing
            if len(cand) > self.n_bins - 1:                       # keep an evenly spaced subset
                cand = cand[np.linspace(0, len(cand) - 1, self.n_bins - 1).round().astype(int)]
            thresholds.append(cand)

        self.max_t = max(1, max(len(t) for t in thresholds))
        self.thresholds = np.full((X.shape[1], self.max_t), np.inf)   # pad with +inf (never chosen)
        for f, t in enumerate(thresholds):
            self.thresholds[f, :len(t)] = t
        return self

    def transform(self, X):
        """bins[f, i] = number of thresholds of feature f below X[i, f]."""
        bins = np.empty((X.shape[1], X.shape[0]), dtype=np.uint8)
        for f in range(X.shape[1]):
            bins[f] = np.searchsorted(self.thresholds[f], X[:, f], side="left")
        return bins                                               # x <= thresholds[f, t]  <=>  bin <= t


# ---------------------------------------------------------------- decision tree
class DecisionTree:
    """CART classification tree (greedy, best impurity decrease at every node)."""

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 criterion="gini", max_features=None, n_bins=64, seed=42):
        self.max_depth = max_depth                    # None = no limit
        self.min_samples_split = min_samples_split    # rows needed to split a node
        self.min_samples_leaf = min_samples_leaf      # rows each child must keep
        self.criterion = criterion                    # "gini" or "entropy"
        self.max_features = max_features              # features tried per node (None = all)
        self.n_bins = n_bins                          # candidate thresholds per feature (+1)
        self.seed = seed

    def fit(self, X, y, n_classes, binner=None, bins=None, rows=None):
        """Train the tree. The forest passes shared binner/bins and bootstrap rows."""
        self.n_classes = n_classes
        if binner is None:
            binner = Binner(self.n_bins).fit(X)
        if bins is None:
            bins = binner.transform(X)
        if rows is None:
            rows = np.arange(len(y))
        self.binner = binner
        self._grow(bins, y, rows)
        return self

    def _grow(self, bins, y, rows):
        impurity = IMPURITY[self.criterion]
        rng = np.random.default_rng(self.seed)
        n_features, C = bins.shape[0], self.n_classes
        n_cells = self.binner.max_t + 1               # bins per feature
        thresholds = self.binner.thresholds

        # the tree is stored as flat lists, one entry per node
        feature, threshold, left, right, value, n_rows, gain_at = [], [], [], [], [], [], []

        def new_node(counts):
            feature.append(-1); threshold.append(0.0); left.append(-1); right.append(-1)
            value.append(counts / counts.sum())       # class shares of the node
            n_rows.append(int(counts.sum())); gain_at.append(0.0)
            return len(feature) - 1

        root_counts = np.bincount(y[rows], minlength=C)
        stack = [(new_node(root_counts), rows, 0, root_counts)]   # (node, row indices, depth, class counts)
        while stack:
            node, idx, depth, counts = stack.pop()
            n = len(idx)

            # stopping rules: the node stays a leaf
            if n < self.min_samples_split or n < 2 * self.min_samples_leaf:
                continue
            if self.max_depth is not None and depth >= self.max_depth:
                continue
            if counts.max() == n:                     # pure node
                continue

            # features this node may look at
            if self.max_features is None or self.max_features >= n_features:
                feats = np.arange(n_features)
            else:
                feats = rng.choice(n_features, self.max_features, replace=False)
            F = len(feats)

            # table[f, b, c] = rows of class c in bin b of feature f (one bincount)
            y_node = y[idx]
            B = bins[feats][:, idx].astype(np.int64)
            cell = (np.arange(F)[:, None] * n_cells + B) * C + y_node[None, :]
            table = np.bincount(cell.ravel(), minlength=F * n_cells * C).reshape(F, n_cells, C)

            # class counts left/right of every threshold, then the gain of every question
            left_counts = np.cumsum(table, axis=1)[:, :-1, :]
            right_counts = counts - left_counts
            n_left = left_counts.sum(axis=2)
            n_right = n - n_left
            after = (n_left * impurity(left_counts) + n_right * impurity(right_counts)) / n
            gain = impurity(counts) - after
            gain[(n_left < self.min_samples_leaf) | (n_right < self.min_samples_leaf)] = -1

            f_i, t = np.unravel_index(np.argmax(gain), gain.shape)
            if gain[f_i, t] <= 1e-12:                 # no question makes the node purer
                continue

            # split the node and push both children
            f = feats[f_i]
            go_left = B[f_i] <= t
            counts_l = left_counts[f_i, t]
            counts_r = counts - counts_l
            feature[node] = int(f)
            threshold[node] = float(thresholds[f, t])
            gain_at[node] = float(gain[f_i, t]) * n   # for feature importance
            left[node] = new_node(counts_l)
            right[node] = new_node(counts_r)
            stack.append((right[node], idx[~go_left], depth + 1, counts_r))
            stack.append((left[node], idx[go_left], depth + 1, counts_l))

        self.feature = np.array(feature, dtype=np.int64)
        self.threshold = np.array(threshold)
        self.left = np.array(left, dtype=np.int64)
        self.right = np.array(right, dtype=np.int64)
        self.value = np.array(value)
        self.n_rows = np.array(n_rows)
        self.weighted_gain = np.array(gain_at)

    def leaf_of(self, X):
        """Leaf reached by every row (all rows go down the tree together)."""
        node = np.zeros(len(X), dtype=np.int64)
        active = np.where(self.feature[node] >= 0)[0]
        while len(active):
            n = node[active]
            go_left = X[active, self.feature[n]] <= self.threshold[n]
            node[active] = np.where(go_left, self.left[n], self.right[n])
            active = active[self.feature[node[active]] >= 0]
        return node

    def predict_proba(self, X):
        return self.value[self.leaf_of(X)]

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    @property
    def n_nodes(self):
        return len(self.feature)

    @property
    def n_leaves(self):
        return int((self.feature < 0).sum())

    def depth(self):
        d = np.zeros(self.n_nodes, dtype=int)
        for n in range(self.n_nodes):                 # parents are stored before children
            if self.feature[n] >= 0:
                d[self.left[n]] = d[self.right[n]] = d[n] + 1
        return int(d.max())

    def feature_importance(self, n_features):
        """Total impurity decrease per feature, normalised to sum to 1."""
        used = self.feature >= 0
        imp = np.bincount(self.feature[used], weights=self.weighted_gain[used], minlength=n_features)
        return imp / max(imp.sum(), 1e-12)


# ---------------------------------------------------------------- random forest
_SHARED = {}    # data given once to every worker process


def _init_worker(bins, y, binner, n_classes):
    _SHARED.update(bins=bins, y=y, binner=binner, n_classes=n_classes)


def _fit_job(job):
    """Train one tree of the forest (runs in a worker process)."""
    params, rows = job
    tree = DecisionTree(**params)
    tree.fit(None, _SHARED["y"], _SHARED["n_classes"],
             binner=_SHARED["binner"], bins=_SHARED["bins"], rows=rows)
    del tree.binner                                   # the forest keeps one copy
    return tree


class RandomForest:
    """Bagged decision trees with random features at every split."""

    def __init__(self, n_trees=100, max_features="sqrt", max_depth=None,
                 min_samples_leaf=1, min_samples_split=2, criterion="gini",
                 bootstrap_fraction=1.0, voting="soft", n_bins=64,
                 seed=42, n_jobs=-1):
        self.n_trees = n_trees
        self.max_features = max_features              # "sqrt", "log2", an int or a fraction
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.bootstrap_fraction = bootstrap_fraction  # rows per tree, as a fraction of n
        self.voting = voting                          # "soft" = average probabilities, "hard" = one vote per tree
        self.n_bins = n_bins
        self.seed = seed
        self.n_jobs = n_jobs                          # -1 = all CPU cores

    def _n_split_features(self, n_features):
        mf = self.max_features
        if mf == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        if mf == "log2":
            return max(1, int(np.log2(n_features)))
        if mf is None:
            return n_features
        if isinstance(mf, float) and mf < 1:
            return max(1, int(mf * n_features))
        return int(mf)

    def fit(self, X, y, n_classes):
        self.n_classes = n_classes
        n_rows, n_features = X.shape
        self.binner = Binner(self.n_bins).fit(X)
        bins = self.binner.transform(X)               # binned once, shared by all trees

        # bootstrap: each tree draws rows with replacement
        rng = np.random.default_rng(self.seed)
        n_draw = int(round(self.bootstrap_fraction * n_rows))
        self.bootstrap_rows = [rng.integers(0, n_rows, n_draw) for _ in range(self.n_trees)]

        params = dict(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                      min_samples_leaf=self.min_samples_leaf, criterion=self.criterion,
                      max_features=self._n_split_features(n_features), n_bins=self.n_bins)
        jobs = [(dict(params, seed=self.seed + i), rows) for i, rows in enumerate(self.bootstrap_rows)]

        # trees are independent, so they are trained in parallel
        workers = os.cpu_count() if self.n_jobs in (-1, None) else self.n_jobs
        if workers == 1:
            _init_worker(bins, y, self.binner, n_classes)
            self.trees = [_fit_job(job) for job in jobs]
        else:
            with ProcessPoolExecutor(workers, initializer=_init_worker,
                                     initargs=(bins, y, self.binner, n_classes)) as pool:
                self.trees = list(pool.map(_fit_job, jobs, chunksize=max(1, len(jobs) // (4 * workers))))
        return self

    def predict_proba(self, X, n_trees=None):
        """Average class probabilities of the first n_trees trees (all by default)."""
        trees = self.trees if n_trees is None else self.trees[:n_trees]
        total = np.zeros((len(X), self.n_classes))
        for tree in trees:
            p = tree.predict_proba(X)
            if self.voting == "hard":
                p = np.eye(self.n_classes)[p.argmax(axis=1)]
            total += p
        return total / len(trees)

    def predict(self, X, n_trees=None):
        return self.predict_proba(X, n_trees).argmax(axis=1)

    def oob_accuracy(self, X, y):
        """Out-of-bag accuracy: each row is predicted only by trees that did not train on it."""
        total = np.zeros((len(X), self.n_classes))
        for tree, rows in zip(self.trees, self.bootstrap_rows):
            oob = np.ones(len(X), dtype=bool)
            oob[rows] = False
            total[oob] += tree.predict_proba(X[oob])
        seen = total.sum(axis=1) > 0
        return float((total[seen].argmax(axis=1) == y[seen]).mean())


# ---------------------------------------------------------------- permutation importance
def permutation_importance(model, X, y, features=None, n_repeats=3, seed=99):
    """Accuracy drop when one feature is shuffled (method from Lab 5)."""
    rng = np.random.default_rng(seed)
    features = np.arange(X.shape[1]) if features is None else np.asarray(features)
    base = (model.predict(X) == y).mean()
    drops = np.zeros(len(features))
    for i, f in enumerate(features):
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, f] = rng.permutation(Xp[:, f])
            drops[i] += base - (model.predict(Xp) == y).mean()
    return features, drops / n_repeats


# ---------------------------------------------------------------- LDA projection
def fit_lda(X, y, n_classes, shrinkage=0.0):
    """Fisher LDA: directions that maximise between-class / within-class spread."""
    with np.errstate(all="ignore"):                   # numpy 2.0 on macOS prints false matmul warnings
        X = X.astype(np.float64)
        d = X.shape[1]
        mean = X.mean(axis=0)
        Sw = np.zeros((d, d))                         # within-class scatter
        Sb = np.zeros((d, d))                         # between-class scatter
        for c in range(n_classes):
            Xc = X[y == c]
            mc = Xc.mean(axis=0)
            Sw += (Xc - mc).T @ (Xc - mc)
            diff = (mc - mean)[:, None]
            Sb += len(Xc) * (diff @ diff.T)
        Sw /= len(X)
        Sb /= len(X)
        Sw = (1 - shrinkage) * Sw + shrinkage * (np.trace(Sw) / d) * np.eye(d)   # shrinkage = regularisation

        # solve Sw^-1 Sb w = lambda w: whiten with Sw, then eigen-decompose
        evals, evecs = np.linalg.eigh(Sw)
        evals = np.maximum(evals, 1e-10 * evals.max())
        W = evecs / np.sqrt(evals)
        lam, V = np.linalg.eigh(W.T @ Sb @ W)
        order = np.argsort(lam)[::-1][:n_classes - 1]   # at most C - 1 useful directions
        return {"mean": mean, "proj": W @ V[:, order], "ratios": lam[order]}


def apply_lda(X, lda):
    """Project rows onto the LDA directions."""
    with np.errstate(all="ignore"):
        return ((X.astype(np.float64) - lda["mean"]) @ lda["proj"]).astype(np.float32)
