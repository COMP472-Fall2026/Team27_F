"""
trees.py -- our from-scratch Decision Tree and Random Forest (NumPy only).

Used by:
    decision_tree/decision_tree.ipynb
    random_forest/random_forest.ipynb

No scikit-learn: every part of the learning algorithm (impurity, split search,
tree growth, bootstrap, voting) is written here.

Contents:
    gini, entropy               impurity of a group of rows
    Binner                      candidate thresholds for every feature
    DecisionTree                CART-style classification tree
    RandomForest                bagged trees with random features per split
    permutation_importance      accuracy drop when a feature is shuffled (Lab 5)
    fit_lda / apply_lda         Linear Discriminant Analysis projection
                                (dimensionality reduction used as preprocessing)

---------------------------------------------------------------------------
How the split search works (the same idea as "histogram" tree libraries)
---------------------------------------------------------------------------
Trying every value of every feature as a threshold is slow. Instead:

1. Before training, for every feature we choose at most `n_bins - 1` candidate
   thresholds (quantiles of that feature).  Each value is then replaced by its
   bin number = how many thresholds are smaller than it.  So
         value <= thresholds[f][t]   is the same as   bin <= t
2. At a node, ONE np.bincount builds the table
         table[f, b, c] = number of rows of class c in bin b of feature f
3. A cumulative sum over the bins gives the class counts on the left side of
   every threshold at once -> the gain of every (feature, threshold) question
   is computed in one vectorised step, and we keep the best one.

Important detail for the image features: 85% of the values are exactly 0.
Plain quantiles would put almost every threshold at 0 (the 1%..85% quantiles
are all 0), leaving only a handful of real choices per feature.  The Binner
therefore also takes quantiles of the NON-ZERO values, and removes duplicates.
"""

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np


# ==========================================================================
# 1. Impurity
# ==========================================================================
# counts[..., c] = number of rows of class c in a group (last axis = classes).
# Both functions return one impurity per group (0 = only one class).

def gini(counts):
    """Gini impurity: 1 - sum_c p_c^2."""
    totals = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(totals, 1)
    return 1.0 - (p ** 2).sum(axis=-1)


def entropy(counts):
    """Entropy: -sum_c p_c * log2(p_c)   (empty classes add nothing)."""
    totals = counts.sum(axis=-1, keepdims=True)
    p = counts / np.maximum(totals, 1)
    log_p = np.log2(np.where(p > 0, p, 1.0))
    return -(p * log_p).sum(axis=-1)


IMPURITY = {"gini": gini, "entropy": entropy}


# ==========================================================================
# 2. Candidate thresholds ("binning")
# ==========================================================================
class Binner:
    """Learns candidate thresholds from TRAINING data only, then turns any
    matrix into bin numbers (uint8, so a 20000 x 500 matrix takes 10 MB)."""

    def __init__(self, n_bins=64):
        assert 2 <= n_bins <= 256, "bins are stored as uint8"
        self.n_bins = n_bins

    def fit(self, X):
        q = np.linspace(0, 1, self.n_bins + 1)[1:-1]          # n_bins - 1 quantile levels
        thresholds = []
        for f in range(X.shape[1]):
            col = X[:, f]
            cand = np.quantile(col, q)
            nonzero = col[col != 0]
            if 0 < len(nonzero) < len(col):                    # zero-inflated feature
                cand = np.concatenate([cand, np.quantile(nonzero, q)])
            cand = np.unique(cand)
            cand = cand[cand < col.max()]                      # a threshold >= max splits nothing
            if len(cand) > self.n_bins - 1:                    # keep an even subset
                keep = np.linspace(0, len(cand) - 1, self.n_bins - 1).round().astype(int)
                cand = cand[keep]
            thresholds.append(cand)

        # Pad to a rectangle with +inf: "x <= inf" sends every row left, so the
        # right side is empty and that question is never chosen.
        self.max_t = max(1, max(len(t) for t in thresholds))
        self.thresholds = np.full((X.shape[1], self.max_t), np.inf)
        for f, t in enumerate(thresholds):
            self.thresholds[f, :len(t)] = t
        return self

    def transform(self, X):
        """bins[f, i] = bin of row i for feature f.  Stored feature-major
        (one contiguous row per feature) because a node reads a few features
        for many rows."""
        bins = np.empty((X.shape[1], X.shape[0]), dtype=np.uint8)
        for f in range(X.shape[1]):
            bins[f] = np.searchsorted(self.thresholds[f], X[:, f], side="left")
        return bins


# ==========================================================================
# 3. Decision Tree
# ==========================================================================
class DecisionTree:
    """CART classification tree.

    Hyper-parameters
        max_depth          None = grow until leaves are pure
        min_samples_split  a node needs at least this many rows to be split
        min_samples_leaf   both children must keep at least this many rows
        criterion          "gini" or "entropy"
        max_features       None = look at all features at every node;
                           an int = look at that many random features (Random Forest)
        n_bins             number of candidate thresholds per feature (+1)

    The fitted tree is stored in flat arrays (one entry per node):
        feature[n]   feature tested at node n (-1 for a leaf)
        threshold[n] go left if x[feature] <= threshold
        left[n], right[n]  children
        value[n, c]  share of training rows of class c in node n
                     (a leaf predicts argmax; a forest averages these)
    """

    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 criterion="gini", max_features=None, n_bins=64, seed=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.n_bins = n_bins
        self.seed = seed

    # ---------------- training ----------------
    def fit(self, X, y, n_classes, binner=None, bins=None, rows=None):
        """Train on X, y.

        binner, bins, rows are optional shortcuts used by the Random Forest:
        the forest bins the data once and every tree reuses it; `rows` are the
        bootstrap row numbers of this tree (they may repeat).
        """
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
        n_features = bins.shape[0]
        C = self.n_classes
        n_cells = self.binner.max_t + 1          # number of bins per feature
        thresholds = self.binner.thresholds

        feature, threshold, left, right, value, n_rows, gain_at = [], [], [], [], [], [], []

        def new_node(counts):
            feature.append(-1); threshold.append(0.0)
            left.append(-1); right.append(-1)
            value.append(counts / counts.sum())
            n_rows.append(int(counts.sum())); gain_at.append(0.0)
            return len(feature) - 1

        # Iterative depth-first growth (a stack instead of recursion, so very
        # deep trees cannot hit Python's recursion limit).
        root_counts = np.bincount(y[rows], minlength=C)
        stack = [(new_node(root_counts), rows, 0, root_counts)]
        while stack:
            node, idx, depth, counts = stack.pop()
            n = len(idx)
            # --- stopping rules -> this node stays a leaf ---
            if n < self.min_samples_split or n < 2 * self.min_samples_leaf:
                continue
            if self.max_depth is not None and depth >= self.max_depth:
                continue
            if counts.max() == n:                             # pure node
                continue

            # --- features this node may use ---
            if self.max_features is None or self.max_features >= n_features:
                feats = np.arange(n_features)
            else:
                feats = rng.choice(n_features, self.max_features, replace=False)
            F = len(feats)

            # --- class histogram per (feature, bin) with ONE bincount ---
            y_node = y[idx]
            B = bins[feats][:, idx].astype(np.int64)          # (F, n)
            cell = (np.arange(F)[:, None] * n_cells + B) * C + y_node[None, :]
            table = np.bincount(cell.ravel(), minlength=F * n_cells * C).reshape(F, n_cells, C)

            # Threshold t sends bins 0..t left -> cumulative sum over bins.
            left_counts = np.cumsum(table, axis=1)[:, :-1, :]  # (F, n_thresholds, C)
            right_counts = counts - left_counts
            n_left = left_counts.sum(axis=2)
            n_right = n - n_left

            after = (n_left * impurity(left_counts) + n_right * impurity(right_counts)) / n
            gain = impurity(counts) - after                    # (F, n_thresholds)
            gain[(n_left < self.min_samples_leaf) | (n_right < self.min_samples_leaf)] = -1

            f_i, t = np.unravel_index(np.argmax(gain), gain.shape)
            if gain[f_i, t] <= 1e-12:                          # nothing makes it purer
                continue

            # --- split ---
            f = feats[f_i]
            go_left = B[f_i] <= t
            idx_l, idx_r = idx[go_left], idx[~go_left]
            counts_l = left_counts[f_i, t]
            counts_r = counts - counts_l
            feature[node] = int(f)
            threshold[node] = float(thresholds[f, t])
            gain_at[node] = float(gain[f_i, t]) * n          # impurity decrease x rows
            left[node] = new_node(counts_l)
            right[node] = new_node(counts_r)
            stack.append((right[node], idx_r, depth + 1, counts_r))
            stack.append((left[node], idx_l, depth + 1, counts_l))

        self.feature = np.array(feature, dtype=np.int64)
        self.threshold = np.array(threshold)
        self.left = np.array(left, dtype=np.int64)
        self.right = np.array(right, dtype=np.int64)
        self.value = np.array(value)
        self.n_rows = np.array(n_rows)
        self.weighted_gain = np.array(gain_at)

    # ---------------- prediction ----------------
    def leaf_of(self, X):
        """Index of the leaf every row ends in. All rows move down the tree
        together, one level per loop iteration (vectorised)."""
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

    # ---------------- size ----------------
    @property
    def n_nodes(self):
        return len(self.feature)

    @property
    def n_leaves(self):
        return int((self.feature < 0).sum())

    def depth(self):
        d = np.zeros(self.n_nodes, dtype=int)
        for n in range(self.n_nodes):            # parents always come before children
            if self.feature[n] >= 0:
                d[self.left[n]] = d[self.right[n]] = d[n] + 1
        return int(d.max())

    def feature_importance(self, n_features):
        """Total (row-weighted) impurity decrease of the splits on each feature,
        normalised to sum to 1 ("mean decrease in impurity")."""
        imp = np.bincount(self.feature[self.feature >= 0],
                          weights=self.weighted_gain[self.feature >= 0], minlength=n_features)
        return imp / max(imp.sum(), 1e-12)


# ==========================================================================
# 4. Random Forest
# ==========================================================================
# Trees are independent, so they are trained in parallel processes.
# Each worker receives the binned data once (initializer), then builds trees.
_SHARED = {}


def _init_worker(bins, y, binner, n_classes):
    _SHARED.update(bins=bins, y=y, binner=binner, n_classes=n_classes)


def _fit_job(job):
    """Build one tree of the forest inside a worker process."""
    params, rows = job
    tree = DecisionTree(**params)
    tree.fit(None, _SHARED["y"], _SHARED["n_classes"],
             binner=_SHARED["binner"], bins=_SHARED["bins"], rows=rows)
    del tree.binner                     # the forest keeps one shared copy
    return tree


class RandomForest:
    """Random Forest = many decision trees that vote.

    The trees are made different in two ways:
      1. bootstrap: every tree trains on n rows drawn WITH replacement
         (about 63% distinct rows; the other ~37% are its "out-of-bag" rows)
      2. random features: every split only looks at `max_features` random features

    Hyper-parameters
        n_trees, max_features ("sqrt", "log2", an int, or a fraction < 1),
        max_depth, min_samples_leaf, min_samples_split, criterion,
        bootstrap_fraction (rows per tree as a fraction of n),
        voting: "soft" = average the trees' class probabilities (default)
                "hard" = one vote per tree
    """

    def __init__(self, n_trees=100, max_features="sqrt", max_depth=None,
                 min_samples_leaf=1, min_samples_split=2, criterion="gini",
                 bootstrap_fraction=1.0, voting="soft", n_bins=64,
                 seed=42, n_jobs=-1):
        self.n_trees = n_trees
        self.max_features = max_features
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.bootstrap_fraction = bootstrap_fraction
        self.voting = voting
        self.n_bins = n_bins
        self.seed = seed
        self.n_jobs = n_jobs

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
        bins = self.binner.transform(X)               # binned ONCE for all trees

        rng = np.random.default_rng(self.seed)
        n_draw = int(round(self.bootstrap_fraction * n_rows))
        self.bootstrap_rows = [rng.integers(0, n_rows, n_draw) for _ in range(self.n_trees)]

        params = dict(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                      min_samples_leaf=self.min_samples_leaf, criterion=self.criterion,
                      max_features=self._n_split_features(n_features), n_bins=self.n_bins)
        jobs = [(dict(params, seed=self.seed + i), rows)          # each tree its own seed
                for i, rows in enumerate(self.bootstrap_rows)]

        workers = os.cpu_count() if self.n_jobs in (-1, None) else self.n_jobs
        if workers == 1:
            _init_worker(bins, y, self.binner, n_classes)
            self.trees = [_fit_job(job) for job in jobs]
        else:
            with ProcessPoolExecutor(workers, initializer=_init_worker,
                                     initargs=(bins, y, self.binner, n_classes)) as pool:
                self.trees = list(pool.map(_fit_job, jobs, chunksize=max(1, len(jobs) // (4 * workers))))
        self._n_train = n_rows
        return self

    def predict_proba(self, X, n_trees=None):
        """Average class probabilities of the first `n_trees` trees
        (or vote shares if voting == "hard")."""
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
        """Out-of-bag accuracy: every training row is predicted only by the
        trees that did NOT see it.  A free validation estimate."""
        total = np.zeros((len(X), self.n_classes))
        for tree, rows in zip(self.trees, self.bootstrap_rows):
            oob = np.ones(len(X), dtype=bool)
            oob[rows] = False
            total[oob] += tree.predict_proba(X[oob])
        seen = total.sum(axis=1) > 0
        return float((total[seen].argmax(axis=1) == y[seen]).mean())


# ==========================================================================
# 5. Permutation importance (as in Lab 5)
# ==========================================================================
def permutation_importance(model, X, y, features=None, n_repeats=3, seed=99):
    """For each feature: shuffle its column, measure how much accuracy drops.
    A big drop means the model relied on that feature.  Returns (features, mean drop)."""
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


# ==========================================================================
# 6. Linear Discriminant Analysis (dimensionality reduction)
# ==========================================================================
def fit_lda(X, y, n_classes, shrinkage=0.0):
    with np.errstate(all="ignore"):     # numpy 2.0 on macOS prints false matmul warnings
        return _fit_lda(X, y, n_classes, shrinkage)


def _fit_lda(X, y, n_classes, shrinkage):
    """Fisher LDA: find the directions w that maximise
            (spread BETWEEN class means) / (spread WITHIN classes).
    There are at most n_classes - 1 such directions.

    shrinkage in [0, 1] mixes the within-class covariance with a multiple of
    the identity:  Sw' = (1 - s) Sw + s * (trace(Sw) / d) I.
    With 768 correlated features Sw is badly conditioned; shrinkage makes the
    projection more stable (it is a regularisation hyper-parameter).

    Returns a dict with the mean to subtract and the projection matrix.
    """
    X = X.astype(np.float64)
    d = X.shape[1]
    mean = X.mean(axis=0)
    Sw = np.zeros((d, d))
    Sb = np.zeros((d, d))
    for c in range(n_classes):
        Xc = X[y == c]
        mc = Xc.mean(axis=0)
        D = Xc - mc
        Sw += D.T @ D
        diff = (mc - mean)[:, None]
        Sb += len(Xc) * (diff @ diff.T)
    Sw /= len(X)
    Sb /= len(X)
    Sw = (1 - shrinkage) * Sw + shrinkage * (np.trace(Sw) / d) * np.eye(d)

    # Solve Sw^-1 Sb w = lambda w by whitening with Sw (numerically stable):
    evals, evecs = np.linalg.eigh(Sw)
    evals = np.maximum(evals, 1e-10 * evals.max())
    W = evecs / np.sqrt(evals)                     # W.T @ Sw @ W = I
    lam, V = np.linalg.eigh(W.T @ Sb @ W)
    order = np.argsort(lam)[::-1][:n_classes - 1]  # biggest ratios first
    return {"mean": mean, "proj": W @ V[:, order], "ratios": lam[order]}


def apply_lda(X, lda):
    """Project rows onto the LDA directions (n_classes - 1 new features)."""
    with np.errstate(all="ignore"):
        return ((X.astype(np.float64) - lda["mean"]) @ lda["proj"]).astype(np.float32)
