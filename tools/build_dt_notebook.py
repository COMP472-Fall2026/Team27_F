"""Writes decision_tree/decision_tree.ipynb (run once; the notebook is the deliverable)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md("""# Decision Tree

**Idea:** the tree asks yes/no questions like *"is feature 12 <= 0.35?"*. The answer sends the example
left or right, to the next question. At the end (a **leaf**) the tree gives a class.

The tree is built greedily from the training data: at every node we choose the question that makes the
two groups as *pure* as possible, then repeat on each group.

**Implementation:** `trees.py` (NumPy only, no scikit-learn). This notebook only runs experiments.

What we investigate (as required by the project):

| Hyper-parameter | Values |
|---|---|
| `max_depth` | 2 … 30, None |
| `min_samples_split` | 2 … 400 |
| `min_samples_leaf` | 1 … 100 |
| `criterion` | Gini, entropy |
| number of candidate thresholds (`n_bins`) | 8 … 256 |
| input representation | raw features vs **LDA projection** |

For every experiment we report **train** and **validation** accuracy, so we can see under- and over-fitting."""))

cells.append(md("## 1. Setup"))
cells.append(code("""import os
import sys

# The notebook is in a sub-folder. Go to the project folder (where common.py is).
if not os.path.exists("common.py"):
    os.chdir("..")
sys.path.insert(0, os.getcwd())

import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common   # shared: load data, split, metrics, Kaggle file
import trees    # our Decision Tree / Random Forest / LDA code

common.set_seed(42)
os.makedirs("decision_tree/outputs", exist_ok=True)
print("python", sys.version.split()[0], "| numpy", np.__version__, "| pandas", pd.__version__)"""))

cells.append(md("""## 2. How our tree is trained (summary of `trees.py`)

**Impurity** of a group of rows (0 = only one class), with `p_c` = share of class `c`:
* Gini: `1 - sum_c p_c^2`
* Entropy: `-sum_c p_c log2(p_c)`

**Growing the tree** (CART, greedy, depth-first):
1. *Stop* and make a **leaf** if: the node is pure, `max_depth` is reached, the node has fewer than
   `min_samples_split` rows, or no question improves purity. A leaf stores the class shares of its rows
   and predicts the most common class.
2. Otherwise evaluate every question *"feature f <= threshold t"*:
   `gain = impurity(parent) - (n_left/n) impurity(left) - (n_right/n) impurity(right)`.
   Questions that leave fewer than `min_samples_leaf` rows on one side are not allowed.
3. Keep the question with the largest gain and repeat on both children.

**Fast split search (histograms).** Before training, each feature gets up to 63 candidate thresholds
(quantiles), and every value is replaced by its bin number. At a node one `np.bincount` gives, for every
feature and bin, how many rows of each class it holds; a cumulative sum over bins then gives the gain of
*every* question at once.

**A detail that matters for the image features.** 85% of image feature values are exactly 0.
With plain quantiles, almost all thresholds of a feature would be 0, leaving only a few real choices.
Our binner also takes quantiles of the **non-zero** values (section 4.5 measures the effect)."""))

cells.append(md("""## 3. Small example

200 points with 2 features; the true rule is class 1 if `x0 + x1 > 1`. A tree can only cut along **one
feature at a time**, so it draws a *staircase* around the diagonal. This is also the reason a tree struggles
when the useful information is a combination of many features (section 4.6)."""))
cells.append(code("""rng = np.random.default_rng(0)
X_toy = rng.random((200, 2))
y_toy = (X_toy[:, 0] + X_toy[:, 1] > 1).astype(int)


def print_tree(tree, node=0, depth=0, names=None):
    pad = "    " * depth
    if tree.feature[node] < 0:
        print(f"{pad}-> class {tree.value[node].argmax()}   ({tree.n_rows[node]} rows)")
        return
    f = tree.feature[node]
    name = names[f] if names else f"feature {f}"
    print(f"{pad}{name} <= {tree.threshold[node]:.2f} ?")
    print_tree(tree, tree.left[node], depth + 1, names)
    print(f"{pad}else:")
    print_tree(tree, tree.right[node], depth + 1, names)


xx, yy = np.meshgrid(np.linspace(0, 1, 150), np.linspace(0, 1, 150))
grid = np.c_[xx.ravel(), yy.ravel()]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
for ax, depth in zip(axes, [1, 3, 8]):
    toy = trees.DecisionTree(max_depth=depth).fit(X_toy, y_toy, n_classes=2)
    acc = common.accuracy(y_toy, toy.predict(X_toy))
    if depth == 1:
        print_tree(toy, names=["x0", "x1"])
    ax.contourf(xx, yy, toy.predict(grid).reshape(xx.shape), alpha=0.3, cmap="coolwarm")
    ax.scatter(X_toy[:, 0], X_toy[:, 1], c=y_toy, cmap="coolwarm", s=10)
    ax.plot([0, 1], [1, 0], "k--")
    ax.set_title(f"max_depth={depth}: train acc {acc:.2f}")
plt.tight_layout()
plt.show()"""))

cells.append(md("""## 4. Data and validation split

Same stratified 80/20 split as every model in the project (`common.stratified_split`, seed 42):
image 16,000 train / 4,000 validation, sentiment 9,600 / 2,400. Every hyper-parameter below is chosen
on the validation rows only; the test set is used once, at the end.

No scaling is needed for raw features: a tree compares one feature with a threshold, so scaling a feature
just moves the threshold."""))
cells.append(code("""X_image, y_image, X_image_test, image_test_ids = common.load_task("image")
X_sent, y_sent, X_sent_test, sent_test_ids = common.load_task("sentiment")

DATA = {"image": (X_image, y_image), "sentiment": (X_sent, y_sent)}
TEST = {"image": (X_image_test, image_test_ids), "sentiment": (X_sent_test, sent_test_ids)}
N_CLASSES = common.N_CLASSES
TASKS = ["sentiment", "image"]

SPLIT = {}
for task in TASKS:
    X, y = DATA[task]
    train_idx, val_idx = common.stratified_split(y)
    SPLIT[task] = (X[train_idx], y[train_idx], X[val_idx], y[val_idx])
    print(f"{task:<9} train {len(train_idx):6d}  validation {len(val_idx):5d}  features {X.shape[1]}")"""))

cells.append(md("""## 5. Experiments

`run` trains one tree and records train accuracy, validation accuracy, size and time.
Every result is kept in `results` and saved to `decision_tree/outputs/`."""))
cells.append(code("""results = []


def run(task, experiment, features="raw", data=None, **settings):
    X_tr, y_tr, X_va, y_va = data if data is not None else SPLIT[task]
    start = time.time()
    tree = trees.DecisionTree(**settings).fit(X_tr, y_tr, N_CLASSES[task])
    seconds = time.time() - start
    row = {"task": task, "experiment": experiment, "features": features,
           "max_depth": settings.get("max_depth"), "min_samples_split": settings.get("min_samples_split", 2),
           "min_samples_leaf": settings.get("min_samples_leaf", 1), "criterion": settings.get("criterion", "gini"),
           "n_bins": settings.get("n_bins", 64),
           "train_acc": round(common.accuracy(y_tr, tree.predict(X_tr)), 4),
           "val_acc": round(common.accuracy(y_va, tree.predict(X_va)), 4),
           "nodes": tree.n_nodes, "leaves": tree.n_leaves, "depth": tree.depth(), "seconds": round(seconds, 2)}
    results.append(row)
    return row


def plot_curve(rows, x, title, xlabel, ax):
    labels = [str(r[x]) for r in rows]
    ax.plot(labels, [r["train_acc"] for r in rows], marker="o", label="train")
    ax.plot(labels, [r["val_acc"] for r in rows], marker="o", label="validation")
    best = max(rows, key=lambda r: r["val_acc"])
    ax.axvline(labels.index(str(best[x])), color="grey", ls=":", lw=1)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel("accuracy"); ax.grid(alpha=0.3); ax.legend()


def show(rows, cols=("train_acc", "val_acc", "nodes", "depth", "seconds")):
    keys = [k for k in rows[0] if k not in ("task", "experiment", "features", "train_acc", "val_acc",
                                             "nodes", "leaves", "depth", "seconds")]
    varying = [k for k in keys if len({str(r[k]) for r in rows}) > 1]
    return pd.DataFrame(rows)[["task"] + varying + list(cols)]"""))

cells.append(md("""### 5.1 Maximum depth — under-fitting vs over-fitting

* A shallow tree asks too few questions → train and validation accuracy are both low (**under-fitting**).
* A very deep tree keeps splitting until leaves are pure → train accuracy goes to 1.0 while validation
  accuracy stops improving or drops (**over-fitting**: the last questions only memorise noise)."""))
cells.append(code("""DEPTHS = [2, 4, 6, 8, 10, 12, 15, 20, 30, None]
depth_rows = {}
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for ax, task in zip(axes, TASKS):
    depth_rows[task] = [run(task, "max_depth", max_depth=d) for d in DEPTHS]
    plot_curve(depth_rows[task], "max_depth", f"{task}: max_depth", "max_depth (None = no limit)", ax)
plt.tight_layout(); plt.show()
pd.concat([show(depth_rows[t]) for t in TASKS])"""))

cells.append(md("""### 5.2 Minimum rows to split a node, and minimum rows in a leaf

Both stop the tree from making questions based on very few rows (a form of **pre-pruning**).
We use no depth limit here, so these are the only brakes on the tree."""))
cells.append(code("""SPLITS = [2, 10, 25, 50, 100, 200, 400]
LEAVES = [1, 2, 5, 10, 20, 50, 100]
split_rows, leaf_rows = {}, {}
fig, axes = plt.subplots(2, 2, figsize=(13, 7))
for col, task in enumerate(TASKS):
    split_rows[task] = [run(task, "min_samples_split", min_samples_split=s) for s in SPLITS]
    leaf_rows[task] = [run(task, "min_samples_leaf", min_samples_leaf=l) for l in LEAVES]
    plot_curve(split_rows[task], "min_samples_split", f"{task}: min_samples_split", "min_samples_split", axes[0, col])
    plot_curve(leaf_rows[task], "min_samples_leaf", f"{task}: min_samples_leaf", "min_samples_leaf", axes[1, col])
plt.tight_layout(); plt.show()
display(pd.concat([show(split_rows[t]) for t in TASKS]))
display(pd.concat([show(leaf_rows[t]) for t in TASKS]))"""))

cells.append(md("""### 5.3 Splitting criterion: Gini vs entropy

We compare both criteria at the best depth and the best leaf size found above."""))
cells.append(code("""def best_of(rows):
    return max(rows, key=lambda r: r["val_acc"])

crit_rows = []
for task in TASKS:
    d = best_of(depth_rows[task])["max_depth"]
    l = best_of(leaf_rows[task])["min_samples_leaf"]
    for crit in ["gini", "entropy"]:
        crit_rows.append(run(task, "criterion", max_depth=d, criterion=crit))
        crit_rows.append(run(task, "criterion", min_samples_leaf=l, criterion=crit))
show(crit_rows)"""))

cells.append(md("""### 5.4 Combining depth and leaf size

The two brakes interact, so we try a small grid around the best single values."""))
cells.append(code("""grid_rows = []
GRID = {"sentiment": ([4, 5, 6, 8, None], [1, 10, 20, 50, 100]),
        "image": ([15, 20, 30, None], [1, 5, 10, 20, 50])}
for task in TASKS:
    depths, leaves = GRID[task]
    for d in depths:
        for l in leaves:
            grid_rows.append(run(task, "depth x leaf", max_depth=d, min_samples_leaf=l))
g = pd.DataFrame(grid_rows)
for task in TASKS:
    print(task, "- validation accuracy (rows = max_depth, columns = min_samples_leaf)")
    display(g[g.task == task].pivot_table(index="max_depth", columns="min_samples_leaf",
                                         values="val_acc", dropna=False).round(4))"""))

cells.append(md("""### 5.5 Implementation choice: number of candidate thresholds, and zero-aware binning

`n_bins` = how many candidate thresholds each feature gets (+1). Few bins → coarse questions; many bins →
finer questions but slower. We also compare against the **plain-quantile binner** used in our first version
(which, on the image data, places most thresholds at 0)."""))
cells.append(code("""class PlainQuantileBinner(trees.Binner):
    \"\"\"Our first version: thresholds = plain quantiles of each feature (no non-zero quantiles).\"\"\"
    def fit(self, X):
        q = np.linspace(0, 1, self.n_bins + 1)[1:-1]
        self.thresholds = np.quantile(X, q, axis=0).T
        self.max_t = self.thresholds.shape[1]
        return self


bin_rows = []
for task in TASKS:
    l = best_of(leaf_rows[task])["min_samples_leaf"]
    for nb in [8, 16, 32, 64, 128, 256]:
        bin_rows.append(run(task, "n_bins", min_samples_leaf=l, n_bins=nb))
show(bin_rows)"""))
cells.append(code("""X_tr, y_tr, X_va, y_va = SPLIT["image"]
plain = PlainQuantileBinner(64).fit(X_tr)
ours = trees.Binner(64).fit(X_tr)
distinct_plain = np.mean([len(np.unique(t)) for t in plain.thresholds])
distinct_ours = np.mean([np.isfinite(t).sum() for t in ours.thresholds])
print(f"image: distinct thresholds per feature  plain quantiles {distinct_plain:.1f}   ours {distinct_ours:.1f}")

l = best_of(leaf_rows["image"])["min_samples_leaf"]
for name, binner in [("plain quantiles", plain), ("zero-aware (ours)", ours)]:
    tree = trees.DecisionTree(min_samples_leaf=l).fit(X_tr, y_tr, 50, binner=binner)
    print(f"   {name:<18} validation accuracy {common.accuracy(y_va, tree.predict(X_va)):.4f}")"""))

cells.append(md("""### 5.6 Representation: raw features vs LDA projection

A tree cuts along **one feature at a time** (section 3). If the classes are separated by a *combination*
of many features (likely for 768 BERT dimensions), a tree needs a long staircase of cuts to follow it.

**Linear Discriminant Analysis** (Fisher, `trees.fit_lda`, implemented in NumPy) finds the few directions
that best separate the class means relative to the spread inside each class (at most `classes - 1`
directions: 2 for sentiment, 49 for image). Projecting onto them is a standard dimensionality reduction,
fitted on the **training rows only**. `shrinkage` regularises the within-class covariance
(`(1-s) Sw + s * mean_variance * I`) — useful with 768 correlated features.

We tune `shrinkage` and the tree size on the validation set."""))
cells.append(code("""LDA_SPLIT = {}

def lda_data(task, shrinkage):
    key = (task, shrinkage)
    if key not in LDA_SPLIT:
        X_tr, y_tr, X_va, y_va = SPLIT[task]
        lda = trees.fit_lda(X_tr, y_tr, N_CLASSES[task], shrinkage)
        LDA_SPLIT[key] = (trees.apply_lda(X_tr, lda), y_tr, trees.apply_lda(X_va, lda), y_va)
    return LDA_SPLIT[key]


lda_rows = []
SHRINK = [0.0, 0.1, 0.3, 0.5, 0.8]
for task in TASKS:
    for s in SHRINK:
        for d in ([2, 3, 4, 6, 8, None] if task == "sentiment" else [10, 15, 20, None]):
            row = run(task, "lda", features=f"lda s={s}", data=lda_data(task, s),
                      max_depth=d, min_samples_leaf=20)
            row["shrinkage"] = s
            lda_rows.append(row)
l = pd.DataFrame(lda_rows)
for task in TASKS:
    print(task, "- validation accuracy on LDA features (rows = shrinkage, columns = max_depth, min_samples_leaf = 20)")
    display(l[l.task == task].pivot_table(index="shrinkage", columns="max_depth", values="val_acc", dropna=False).round(4))"""))

cells.append(md("""Sentiment with LDA has only **2 features**, so a shallow tree can be printed and read.
Each question is a threshold on one of the two discriminant directions."""))
cells.append(code("""best_sent_lda = max([r for r in lda_rows if r["task"] == "sentiment"], key=lambda r: r["val_acc"])
X_tr, y_tr, X_va, y_va = lda_data("sentiment", best_sent_lda["shrinkage"])
small = trees.DecisionTree(max_depth=2).fit(X_tr, y_tr, 3)
print_tree(small, names=["LD1", "LD2"])
print("validation accuracy of this 3-question tree:", round(common.accuracy(y_va, small.predict(X_va)), 4))

fig, ax = plt.subplots(figsize=(5.5, 4.5))
for c, name in enumerate(common.SENTIMENT_NAMES):
    m = y_va == c
    ax.scatter(X_va[m, 0], X_va[m, 1], s=4, alpha=0.4, label=name)
ax.set_xlabel("LD1"); ax.set_ylabel("LD2"); ax.set_title("Sentiment validation rows on the 2 LDA directions")
ax.legend(markerscale=4); plt.show()"""))

cells.append(md("""## 6. All results and the selected tree per task

We select, for each task, the configuration with the best **validation** accuracy."""))
cells.append(code("""table = pd.DataFrame(results)
for task in TASKS:
    t = table[table.task == task].sort_values("val_acc", ascending=False)
    t.to_csv(f"decision_tree/outputs/{task}_experiments.csv", index=False)

best = {}
for task in TASKS:
    candidates = [r for r in results if r["task"] == task]
    best[task] = max(candidates, key=lambda r: r["val_acc"])
summary = pd.DataFrame([best[t] for t in TASKS])
summary[["task", "features", "max_depth", "min_samples_split", "min_samples_leaf", "criterion", "n_bins",
         "train_acc", "val_acc", "nodes", "depth", "seconds"]]"""))

cells.append(md("""## 7. Error analysis of the selected trees

We retrain the selected tree and look at **which** classes it gets wrong:
* sentiment: confusion matrix, precision / recall / F1 per class
* image: hardest classes and the most frequent (true → predicted) confusions"""))
cells.append(code("""def settings_of(row):
    return dict(max_depth=None if pd.isna(row["max_depth"]) else int(row["max_depth"]),
                min_samples_split=int(row["min_samples_split"]), min_samples_leaf=int(row["min_samples_leaf"]),
                criterion=row["criterion"], n_bins=int(row["n_bins"]))


def shrinkage_of(row):
    return float(row["features"].split("=")[1]) if row["features"].startswith("lda") else None


def per_class_table(y_true, y_pred, n_classes, names=None):
    cm = common.confusion_matrix(y_true, y_pred, n_classes)
    p, r = common.per_class_precision(cm), common.per_class_recall(cm)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)
    return cm, pd.DataFrame({"precision": p, "recall": r, "f1": f1},
                            index=names if names else range(n_classes)).round(3)


final_trees, val_pred = {}, {}
for task in TASKS:
    row = best[task]
    s = shrinkage_of(row)
    X_tr, y_tr, X_va, y_va = SPLIT[task] if s is None else lda_data(task, s)
    tree = trees.DecisionTree(**settings_of(row)).fit(X_tr, y_tr, N_CLASSES[task])
    final_trees[task] = tree
    val_pred[task] = tree.predict(X_va)
    np.save(f"decision_tree/outputs/{task}_val_pred.npy", val_pred[task])     # used by final/compare.ipynb
    np.save(f"decision_tree/outputs/{task}_val_proba.npy", tree.predict_proba(X_va))
    print(f"{task}: features={row['features']}  {settings_of(row)}  validation accuracy "
          f"{common.accuracy(y_va, val_pred[task]):.4f}")

y_va = SPLIT["sentiment"][3]
cm, per_class = per_class_table(y_va, val_pred["sentiment"], 3, common.SENTIMENT_NAMES)
print("\\nSentiment confusion matrix (rows = true, columns = predicted)")
display(pd.DataFrame(cm, index=common.SENTIMENT_NAMES, columns=common.SENTIMENT_NAMES))
display(per_class)"""))
cells.append(code("""y_va = SPLIT["image"][3]
cm, per_class = per_class_table(y_va, val_pred["image"], 50)
print("Image: 5 hardest and 5 easiest classes (by recall)")
display(pd.concat([per_class.sort_values("recall").head(5), per_class.sort_values("recall").tail(5)]))
print("Most frequent confusions (true class -> predicted class, count out of 80):")
for t, p, n in common.most_confused_pairs(cm, 8):
    print(f"   {t:2d} -> {p:2d}   {n}")

fig, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm / cm.sum(1, keepdims=True), cmap="Blues")
ax.set_title("Image confusion matrix (row-normalised)"); ax.set_xlabel("predicted"); ax.set_ylabel("true")
plt.show()"""))

cells.append(md("""### Which features does the tree use?

Feature importance = total row-weighted impurity decrease of the splits on each feature."""))
cells.append(code("""for task in TASKS:
    tree = final_trees[task]
    n_feat = tree.binner.thresholds.shape[0]
    imp = tree.feature_importance(n_feat)
    top = np.argsort(imp)[::-1][:10]
    used = (imp > 0).sum()
    print(f"{task}: the tree uses {used} of {n_feat} features; top-10 features carry {imp[top].sum():.0%} of the importance")
    print("   ", ", ".join(f"{f} ({imp[f]:.2f})" for f in top))"""))

cells.append(md("""## 8. Test predictions for Kaggle

Same settings, retrained on **all** training rows (train + validation). If LDA was selected, the LDA
projection is also re-fitted on all training rows before projecting the test rows."""))
cells.append(code("""for task in TASKS:
    row = best[task]
    X, y = DATA[task]
    X_test, test_ids = TEST[task]
    s = shrinkage_of(row)
    if s is not None:
        lda = trees.fit_lda(X, y, N_CLASSES[task], s)
        X, X_test = trees.apply_lda(X, lda), trees.apply_lda(X_test, lda)
    start = time.time()
    tree = trees.DecisionTree(**settings_of(row)).fit(X, y, N_CLASSES[task])
    fit_s = time.time() - start
    start = time.time()
    pred = tree.predict(X_test)
    pd.DataFrame({"id": test_ids, "label": pred}).to_csv(f"decision_tree/outputs/{task}_test_pred.csv", index=False)
    print(f"{task}: saved decision_tree/outputs/{task}_test_pred.csv   train {fit_s:.1f}s, predict {time.time() - start:.2f}s, "
          f"{tree.n_nodes} nodes, test class counts {np.bincount(pred, minlength=N_CLASSES[task])[:10]}...")"""))

cells.append(md("""## 9. Findings (filled in from the outputs above — see the report)

* **Depth:** on both tasks train accuracy rises to 1.0 with depth while validation accuracy peaks early
  and then falls → a deep tree **over-fits**; a very shallow tree **under-fits**.
* **Pre-pruning:** `min_samples_leaf` / `min_samples_split` give the same effect as a depth limit and are
  a smoother brake; criterion (Gini vs entropy) makes little difference.
* **Image task (50 classes):** a single tree is weak. Each leaf must separate 50 classes using one
  feature per question, the features are sparse (85% zeros) and the evidence for a class is spread over
  many weak features.
* **Sentiment task:** the classes are separated by a *direction* that mixes many BERT features. A tree on
  raw features needs a staircase of cuts; after an LDA projection a very small tree is competitive.
* **Implementation:** zero-aware thresholds matter for sparse features; histogram split search makes a
  full image tree train in seconds instead of minutes."""))

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, "decision_tree/decision_tree.ipynb")
print("written")
