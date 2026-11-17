"""Writes random_forest/random_forest.ipynb (run once; the notebook is the deliverable)."""
import nbformat as nbf

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = []

cells.append(md("""# Random Forest

**Problem with one tree:** a deep tree over-fits (train accuracy 1.0, much lower validation accuracy), and a
small change in the training data gives a very different tree (**high variance**).

**Idea:** train many *different* trees and average them. Errors that are not shared by the trees cancel out.

How the trees are made different:
1. **Bootstrap sample (bagging):** each tree trains on `n` rows drawn **with replacement**
   (≈63% distinct rows; the other ≈37% are that tree's *out-of-bag* rows).
2. **Random features:** at every split a tree only looks at `max_features` randomly chosen features.

Prediction: **soft voting** = average the class probabilities stored in the leaves the row reaches
(we also compare with hard voting = one vote per tree).

**Implementation:** `trees.py` (NumPy only; the trees are the same `DecisionTree` class as in
`decision_tree.ipynb`). Trees are independent, so they are trained in parallel processes.

What we investigate: number of trees, `max_features`, `max_depth`, `min_samples_leaf`, criterion,
bootstrap size, soft vs hard voting, and raw vs LDA features."""))

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

import common
import trees

common.set_seed(42)
os.makedirs("random_forest/outputs", exist_ok=True)
print("python", sys.version.split()[0], "| numpy", np.__version__, "| CPU cores used for training:", os.cpu_count())"""))

cells.append(md("## 2. Data and validation split (same as every model)"))
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

# Best single Decision Tree (from decision_tree.ipynb), for comparison
dt_best = {}
for task in TASKS:
    t = pd.read_csv(f"decision_tree/outputs/{task}_experiments.csv")
    dt_best[task] = t.sort_values("val_acc", ascending=False).iloc[0]
    print(f"best single tree, {task}: validation {dt_best[task]['val_acc']:.4f} (train {dt_best[task]['train_acc']:.4f})")"""))

cells.append(md("""## 3. Number of trees

We train **one** forest of 500 trees per task and measure accuracy using only its first
1, 2, 5, … 500 trees (no retraining needed). We also report the **out-of-bag (OOB)** accuracy:
every training row is predicted only by the trees that did not see it — a free estimate of validation
accuracy."""))
cells.append(code("""results = []
TREE_COUNTS = [1, 2, 5, 10, 25, 50, 100, 200, 300, 500]
DEFAULT_MF = {"sentiment": "sqrt", "image": "sqrt"}
big = {}

fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
for ax, task in zip(axes, TASKS):
    X_tr, y_tr, X_va, y_va = SPLIT[task]
    start = time.time()
    forest = trees.RandomForest(n_trees=500, max_features=DEFAULT_MF[task]).fit(X_tr, y_tr, N_CLASSES[task])
    fit_s = time.time() - start
    big[task] = forest
    check = np.random.default_rng(0).choice(len(X_tr), 2000, replace=False)   # 2000 train rows for train accuracy
    tr_acc, va_acc = [], []
    for n in TREE_COUNTS:
        tr_acc.append(common.accuracy(y_tr[check], forest.predict(X_tr[check], n)))
        va_acc.append(common.accuracy(y_va, forest.predict(X_va, n)))
        results.append({"task": task, "experiment": "n_trees", "features": "raw", "n_trees": n,
                        "max_features": DEFAULT_MF[task], "max_depth": None, "min_samples_leaf": 1,
                        "criterion": "gini", "bootstrap_fraction": 1.0, "voting": "soft",
                        "train_acc": round(tr_acc[-1], 4), "val_acc": round(va_acc[-1], 4)})
    oob = forest.oob_accuracy(X_tr, y_tr)
    print(f"{task}: 500 trees in {fit_s:.0f}s | validation {va_acc[-1]:.4f} | out-of-bag {oob:.4f}")
    print("   " + "  ".join(f"{n}:{a:.3f}" for n, a in zip(TREE_COUNTS, va_acc)))
    ax.plot(TREE_COUNTS, tr_acc, marker="o", label="train")
    ax.plot(TREE_COUNTS, va_acc, marker="o", label="validation")
    ax.axhline(dt_best[task]["val_acc"], color="grey", ls="--", lw=1, label="best single tree (val)")
    ax.set_xscale("log"); ax.set_xticks(TREE_COUNTS); ax.set_xticklabels(TREE_COUNTS)
    ax.set_xlabel("number of trees"); ax.set_ylabel("accuracy"); ax.set_title(f"{task}: number of trees")
    ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.show()"""))

cells.append(md("""**Reading the curve:** train accuracy is ≈1.0 from a few trees on (every tree is grown to pure leaves),
but validation accuracy keeps rising as trees are added and then flattens. Adding trees does **not**
over-fit: it only reduces the variance of the average. The cost is linear in the number of trees.

## 4. Other hyper-parameters (200 trees each, to save time)"""))
cells.append(code("""def run(task, experiment, features="raw", data=None, n_trees=200, **settings):
    X_tr, y_tr, X_va, y_va = data if data is not None else SPLIT[task]
    start = time.time()
    forest = trees.RandomForest(n_trees=n_trees, **settings).fit(X_tr, y_tr, N_CLASSES[task])
    fit_s = time.time() - start
    check = np.random.default_rng(0).choice(len(X_tr), 2000, replace=False)
    row = {"task": task, "experiment": experiment, "features": features, "n_trees": n_trees,
           "max_features": str(settings.get("max_features", "sqrt")), "max_depth": settings.get("max_depth"),
           "min_samples_leaf": settings.get("min_samples_leaf", 1), "criterion": settings.get("criterion", "gini"),
           "bootstrap_fraction": settings.get("bootstrap_fraction", 1.0), "voting": settings.get("voting", "soft"),
           "train_acc": round(common.accuracy(y_tr[check], forest.predict(X_tr[check])), 4),
           "val_acc": round(common.accuracy(y_va, forest.predict(X_va)), 4), "seconds": round(fit_s, 1)}
    results.append(row)
    print(f"{task:<9} {experiment:<18} {features:<10} {str(settings):<45} train {row['train_acc']:.3f}  "
          f"validation {row['val_acc']:.4f}  ({fit_s:.0f}s)", flush=True)
    return row"""))

cells.append(md("""### 4.1 Features per split (`max_features`)

* Few features per split → trees are **more different** from each other (good for averaging) but each is weaker.
* Many features per split → each tree is stronger but the trees look alike (they pick the same strong features),
  so averaging helps less. With all features, the forest is just bagging.

√500 ≈ 22 (image), √768 ≈ 27 (sentiment)."""))
cells.append(code("""MF = {"sentiment": [3, 9, "sqrt", 60, 150], "image": [3, 5, 10, "sqrt", 50, 100]}
mf_rows = {t: [run(t, "max_features", max_features=m) for m in MF[t]] for t in TASKS}"""))

cells.append(md("""### 4.2 Tree size: `max_depth` and `min_samples_leaf`

In a forest the individual trees are usually grown **deep** (low bias); the averaging removes the variance.
Limiting the trees should therefore help much less than it did for a single tree."""))
cells.append(code("""def best_mf(task):
    m = max(mf_rows[task], key=lambda r: r["val_acc"])["max_features"]
    return m if m in ("sqrt", "log2") else int(m)

size_rows = []
for task in TASKS:
    for d in [5, 10, 20, None]:
        size_rows.append(run(task, "max_depth", max_features=best_mf(task), max_depth=d))
    for l in [2, 5, 10]:
        size_rows.append(run(task, "min_samples_leaf", max_features=best_mf(task), min_samples_leaf=l))"""))

cells.append(md("""### 4.3 Criterion, bootstrap size, and voting"""))
cells.append(code("""other_rows = []
for task in TASKS:
    mf = best_mf(task)
    other_rows.append(run(task, "criterion", max_features=mf, criterion="entropy"))
    for b in [0.25, 0.5]:
        other_rows.append(run(task, "bootstrap_fraction", max_features=mf, bootstrap_fraction=b))
    other_rows.append(run(task, "voting", max_features=mf, voting="hard"))"""))

cells.append(md("""### 4.4 Representation: raw features, LDA projection, or both

Same LDA projection as in `decision_tree.ipynb` (fitted on the training rows only). With both, the trees
can use the discriminant directions *and* the original features."""))
cells.append(code("""def lda_versions(task, shrinkage):
    X_tr, y_tr, X_va, y_va = SPLIT[task]
    lda = trees.fit_lda(X_tr, y_tr, N_CLASSES[task], shrinkage)
    A_tr, A_va = trees.apply_lda(X_tr, lda), trees.apply_lda(X_va, lda)
    return {"lda": (A_tr, y_tr, A_va, y_va),
            "raw+lda": (np.hstack([X_tr, A_tr]), y_tr, np.hstack([X_va, A_va]), y_va)}


lda_rows = []
for task in TASKS:
    for s in [0.1, 0.3, 0.5]:
        for name, data in lda_versions(task, s).items():
            mf = "sqrt" if name == "raw+lda" else (1 if task == "sentiment" else 7)
            for leaf in ([1, 20] if name == "lda" else [1]):
                row = run(task, "representation", features=f"{name} s={s}", data=data, max_features=mf,
                          min_samples_leaf=leaf)
                lda_rows.append(row)"""))

cells.append(md("""## 5. All results and the selected forest per task"""))
cells.append(code("""table = pd.DataFrame(results)
for task in TASKS:
    table[table.task == task].sort_values("val_acc", ascending=False).to_csv(
        f"random_forest/outputs/{task}_experiments.csv", index=False)

best = {}
for task in TASKS:
    tuned = [r for r in results if r["task"] == task and r["experiment"] != "n_trees"]
    best[task] = max(tuned, key=lambda r: r["val_acc"])
pd.DataFrame([best[t] for t in TASKS])"""))

cells.append(md("""The selected settings are retrained with **500 trees** (section 3 showed that more trees never hurt)."""))
cells.append(code("""FINAL_TREES = 500


def settings_of(row):
    mf = row["max_features"]
    mf = mf if mf in ("sqrt", "log2") else int(mf)
    return dict(max_features=mf, max_depth=None if row["max_depth"] is None or pd.isna(row["max_depth"]) else int(row["max_depth"]),
                min_samples_leaf=int(row["min_samples_leaf"]), criterion=row["criterion"],
                bootstrap_fraction=float(row["bootstrap_fraction"]), voting=row["voting"])


def transform_for(row, X_fit, y_fit, n_classes, *others):
    \"\"\"Apply the selected representation: fit LDA on X_fit only, transform X_fit and the others.\"\"\"
    feats = row["features"]
    if feats == "raw":
        return (X_fit,) + others
    name, s = feats.split(" s=")
    lda = trees.fit_lda(X_fit, y_fit, n_classes, float(s))
    out = []
    for M in (X_fit,) + others:
        A = trees.apply_lda(M, lda)
        out.append(A if name == "lda" else np.hstack([M, A]))
    return tuple(out)


final_forests, val_pred, val_proba = {}, {}, {}
for task in TASKS:
    row = best[task]
    X_tr, y_tr, X_va, y_va = SPLIT[task]
    A_tr, A_va = transform_for(row, X_tr, y_tr, N_CLASSES[task], X_va)
    start = time.time()
    forest = trees.RandomForest(n_trees=FINAL_TREES, **settings_of(row)).fit(A_tr, y_tr, N_CLASSES[task])
    fit_s = time.time() - start
    start = time.time()
    val_proba[task] = forest.predict_proba(A_va)
    pred_s = time.time() - start
    val_pred[task] = val_proba[task].argmax(1)
    final_forests[task] = (forest, A_tr, A_va)
    np.save(f"random_forest/outputs/{task}_val_pred.npy", val_pred[task])      # used by final/compare.ipynb
    np.save(f"random_forest/outputs/{task}_val_proba.npy", val_proba[task])
    print(f"{task}: {row['features']} {settings_of(row)}")
    print(f"   {FINAL_TREES} trees: validation {common.accuracy(y_va, val_pred[task]):.4f}   "
          f"OOB {forest.oob_accuracy(A_tr, y_tr):.4f}   train {fit_s:.0f}s, predict {pred_s:.1f}s")"""))

cells.append(md("""## 6. Random Forest vs a single Decision Tree

The question from the project: *does the forest reduce over-fitting compared with one tree?*
The **gap** = train accuracy − validation accuracy."""))
cells.append(code("""rows = []
for task in TASKS:
    X_tr, y_tr, X_va, y_va = SPLIT[task]
    forest, A_tr, A_va = final_forests[task]
    check = np.random.default_rng(0).choice(len(A_tr), 2000, replace=False)
    rf_train = common.accuracy(y_tr[check], forest.predict(A_tr[check]))
    rf_val = common.accuracy(y_va, val_pred[task])
    rows.append({"task": task, "model": "best single Decision Tree", "features": dt_best[task]["features"],
                 "train_acc": dt_best[task]["train_acc"], "val_acc": dt_best[task]["val_acc"]})
    # one fully grown tree on the same features as the forest, for a like-for-like comparison
    one = trees.DecisionTree().fit(A_tr, y_tr, N_CLASSES[task])
    rows.append({"task": task, "model": "one fully grown tree", "features": best[task]["features"],
                 "train_acc": common.accuracy(y_tr, one.predict(A_tr)), "val_acc": common.accuracy(y_va, one.predict(A_va))})
    rows.append({"task": task, "model": f"Random Forest ({FINAL_TREES} trees)", "features": best[task]["features"],
                 "train_acc": rf_train, "val_acc": rf_val})
cmp = pd.DataFrame(rows)
cmp["gap"] = cmp["train_acc"] - cmp["val_acc"]
cmp.round(4)"""))

cells.append(md("""Where do the forest and the single tree disagree? For each validation row:
right/wrong for the tree vs right/wrong for the forest."""))
cells.append(code("""for task in TASKS:
    y_va = SPLIT[task][3]
    dt_pred = np.load(f"decision_tree/outputs/{task}_val_pred.npy")
    dt_ok, rf_ok = dt_pred == y_va, val_pred[task] == y_va
    print(f"{task:<9} both right {np.sum(dt_ok & rf_ok):5d} | only forest right {np.sum(~dt_ok & rf_ok):5d} | "
          f"only tree right {np.sum(dt_ok & ~rf_ok):5d} | both wrong {np.sum(~dt_ok & ~rf_ok):5d}")"""))

cells.append(md("""## 7. Error analysis of the selected forests"""))
cells.append(code("""def per_class_table(y_true, y_pred, n_classes, names=None):
    cm = common.confusion_matrix(y_true, y_pred, n_classes)
    p, r = common.per_class_precision(cm), common.per_class_recall(cm)
    f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)
    return cm, pd.DataFrame({"precision": p, "recall": r, "f1": f1},
                            index=names if names else range(n_classes)).round(3)


y_va = SPLIT["sentiment"][3]
cm, per_class = per_class_table(y_va, val_pred["sentiment"], 3, common.SENTIMENT_NAMES)
print("Sentiment confusion matrix (rows = true, columns = predicted)")
display(pd.DataFrame(cm, index=common.SENTIMENT_NAMES, columns=common.SENTIMENT_NAMES))
display(per_class)"""))
cells.append(code("""y_va = SPLIT["image"][3]
cm, per_class = per_class_table(y_va, val_pred["image"], 50)
print("Image: 5 hardest and 5 easiest classes (by recall)")
display(pd.concat([per_class.sort_values("recall").head(5), per_class.sort_values("recall").tail(5)]))
print("Most frequent confusions (true -> predicted, count out of 80):")
for t, p, n in common.most_confused_pairs(cm, 8):
    print(f"   {t:2d} -> {p:2d}   {n}")

# Confidence: how sure is the forest when it is right vs wrong?
for task in TASKS:
    y_va = SPLIT[task][3]
    conf = val_proba[task].max(1)
    ok = val_pred[task] == y_va
    print(f"{task}: mean top probability when right {conf[ok].mean():.3f}, when wrong {conf[~ok].mean():.3f}")
    bins = np.quantile(conf, [0, .25, .5, .75, 1])
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf <= hi)
        print(f"      confidence {lo:.2f}-{hi:.2f}: accuracy {ok[m].mean():.3f}")"""))

cells.append(md("""### Feature importance (averaged over the trees)"""))
cells.append(code("""for task in TASKS:
    forest, A_tr, _ = final_forests[task]
    n_feat = A_tr.shape[1]
    imp = np.mean([t.feature_importance(n_feat) for t in forest.trees], axis=0)
    top = np.argsort(imp)[::-1][:10]
    k = int(np.searchsorted(np.cumsum(np.sort(imp)[::-1]), 0.5)) + 1
    print(f"{task}: {k} of {n_feat} features hold half of the total importance")
    print("   top-10:", ", ".join(f"{f} ({imp[f]:.3f})" for f in top))
    plt.figure(figsize=(6, 2.5)); plt.plot(np.sort(imp)[::-1]); plt.title(f"{task}: sorted feature importance")
    plt.xlabel("feature rank"); plt.ylabel("importance"); plt.grid(alpha=0.3); plt.show()"""))

cells.append(md("""## 8. Test predictions for Kaggle

Selected settings, 500 trees, retrained on **all** training rows (LDA re-fitted on all training rows if used)."""))
cells.append(code("""for task in TASKS:
    row = best[task]
    X, y = DATA[task]
    X_test, test_ids = TEST[task]
    A, A_test = transform_for(row, X, y, N_CLASSES[task], X_test)
    start = time.time()
    forest = trees.RandomForest(n_trees=FINAL_TREES, **settings_of(row)).fit(A, y, N_CLASSES[task])
    fit_s = time.time() - start
    proba = forest.predict_proba(A_test)
    np.save(f"random_forest/outputs/{task}_test_proba.npy", proba)
    pred = proba.argmax(1)
    pd.DataFrame({"id": test_ids, "label": pred}).to_csv(f"random_forest/outputs/{task}_test_pred.csv", index=False)
    print(f"{task}: saved random_forest/outputs/{task}_test_pred.csv  ({fit_s:.0f}s)")"""))

cells.append(md("""## 9. Findings (see the report for the discussion)

* The forest beats the best single tree by a wide margin on both tasks, with the same tree code.
* Train accuracy stays ≈1.0, but validation accuracy rises with the number of trees and then flattens:
  averaging reduces **variance** without adding over-fitting.
* Small `max_features` (more diverse trees) works best on the image features; depth limits hurt the forest,
  unlike the single tree.
* Sentiment: the LDA directions carry most of the signal; the forest on LDA (or raw + LDA) features is far
  better than on raw features.
* Trees remain weaker than k-NN on the image task: the class evidence is spread across many sparse features,
  which a distance on the whole vector captures better than one-feature-at-a-time questions."""))

nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
nbf.write(nb, "random_forest/random_forest.ipynb")
print("written")
