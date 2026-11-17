# COMP 472 Project — Team 27 (section F)

Kaggle competition: [COMP472 Project - Fall 2026](https://www.kaggle.com/competitions/comp-472-project-fall-2026-concordia)
(Kaggle team `COMP472_GROUP27_F`).

Two classification tasks on **pre-extracted feature vectors**:

| Task | Classes | Features | Train rows | Test rows |
|---|---|---|---|---|
| Image (visual object recognition) | 50 | 500 (sparse, ≥ 0) | 20,000 | 5,000 |
| Sentiment (BERT mean-pooled embeddings) | 3 (Negative / Neutral / Positive) | 768 | 12,000 | 5,000 |

Score = (image accuracy + sentiment accuracy) / 2.

We implement and compare k-NN, Decision Tree, Random Forest and a feed-forward Neural Network,
plus a k-Means clustering study. k-NN, Decision Tree, Random Forest and k-Means are written from scratch
with NumPy; the Neural Network uses PyTorch (allowed by the project rules).

## Repository contents

| Path | Purpose |
|---|---|
| `common.py` | Shared code: data loading, the **stratified 80/20 train/validation split** (seed 42) used by every model, preprocessing (standardize, sqrt + L2), metrics, Kaggle submission writer with safety checks |
| `trees.py` | From-scratch **Decision Tree** (CART, Gini/entropy, histogram split search), **Random Forest** (bootstrap, random features per split, soft/hard voting, out-of-bag accuracy, parallel training) and **LDA** projection (Fisher LDA with shrinkage, used as dimensionality reduction) |
| `knn/knn.ipynb` | k-NN experiments: k, distance (Euclidean, Manhattan, cosine, χ²), normalisation, distance-weighted voting |
| `decision_tree/decision_tree.ipynb` | Decision Tree experiments: max_depth, min_samples_split, min_samples_leaf, criterion, number of thresholds, raw vs LDA features; error analysis |
| `random_forest/random_forest.ipynb` | Random Forest experiments: number of trees, max_features, max_depth, min_samples_leaf, criterion, bootstrap size, voting, raw vs LDA features; forest vs single tree; error analysis |
| `neural_network/neural_network.ipynb` | Neural Network (PyTorch MLP) experiments: hidden layers/sizes, activation, learning rate, optimizer, batch size, dropout, preprocessing |
| `clustering/clustering.ipynb` | k-Means clustering study (purity, ARI, NMI vs number of clusters) |
| `final/compare.ipynb` | Reads every model's validation predictions → comparison table, error overlap between models, and one Kaggle file per (image model, sentiment model) pair ranked by validation score |
| `final/ensemble.ipynb` | **Soft-voting ensemble**: averages the class probabilities of k-NN, Neural Network (5 seeds) and Random Forest with weights chosen on the validation split (with a held-out-half stability check) → `final/submissions/ensemble_soft_voting.csv` |
| `final/summary.ipynb` | Read-only summary of the generated submission files |
| `<model>/outputs/` | Saved experiment tables (`*_experiments.csv`), validation predictions (`*_val_pred.npy`) and test predictions (`*_test_pred.csv`) |
| `final/submissions/` | Generated Kaggle files and `expected_scores.csv` |
| `tools/build_*_notebook.py` | Scripts that generate the Decision Tree / Random Forest notebooks (the notebooks themselves are the deliverable) |
| `requirements.txt` | Library versions we used |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Download the competition data from the Kaggle **Data** tab ("Download All") and unzip it into
`Project_info/` at the project root (this folder is git-ignored because the files are large):

```
Project_info/
    image_train.csv  image_test.csv
    sentiment_train.csv  sentiment_test.csv
    sample_submission.csv
```

## How to run

No separate pre-processing step is needed: each notebook loads the CSVs through `common.load_task`,
makes the same validation split with `common.stratified_split`, and applies its own preprocessing,
always fitted on the training rows only.

1. Run the four model notebooks (any order); each trains, tunes on the validation split, analyses errors,
   then retrains on all training rows and writes `<model>/outputs/{task}_test_pred.csv`:
   ```bash
   jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 knn/knn.ipynb
   jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 decision_tree/decision_tree.ipynb
   jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 random_forest/random_forest.ipynb
   jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 neural_network/neural_network.ipynb
   ```
   `random_forest.ipynb` reads the Decision Tree results for its comparison, so run it after `decision_tree.ipynb`.
2. Run `clustering/clustering.ipynb` (independent of the others).
3. Run `final/compare.ipynb` → `final/submissions/*.csv`; the first row of `expected_scores.csv` is the best
   single-model pair. Then `final/summary.ipynb`.
4. Run `final/ensemble.ipynb` (needs `random_forest.ipynb` to have run; ≈ 5 min) →
   `final/submissions/ensemble_soft_voting.csv`, our final Kaggle file.

Approximate run times on a laptop CPU: Decision Tree ≈ 17 min (it runs ~150 tree fits),
Random Forest ≈ 20 min (multi-core).

## Results (validation accuracy, same split for every model)

| Method | Image | Sentiment |
|---|---|---|
| k-NN | 0.601 | 0.600 |
| Decision Tree | 0.168 | 0.669 |
| Random Forest | 0.369 | 0.671 |
| Neural Network | 0.457 | 0.687 |
| **Soft-voting ensemble (k-NN + NN + RF)** | **0.636** | **0.688** |

Expected Kaggle score: best single models 0.644 → ensemble 0.662.

## Reproducibility

* Python 3.9.6; library versions in `requirements.txt`.
* Seed 42 everywhere (`common.set_seed`, the split, every tree/forest seed; forest tree *i* uses seed 42 + *i*).
* All hyper-parameters are chosen on the validation split only. The test set is used once, to write the Kaggle files.
