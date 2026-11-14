# COMP472 Project – Fall 2026

Kaggle competition: https://www.kaggle.com/competitions/comp-472-project-fall-2026-concordia

Two classification tasks on pre-extracted feature vectors:

- **Image** – 50 classes, 500 features
- **Sentiment** – 3 classes, 768 features (frozen BERT)

## Folder structure

```
common.py                  shared by everyone: load data, 80/20 split, preprocessing, metrics, Kaggle file
knn/
    knn.ipynb              k-NN from scratch (new version with chi-squared distance, NOT run yet)
    knn_v1_results.ipynb   first k-NN run, with all outputs
neural_network/
    neural_network.ipynb   feed-forward network (PyTorch), with outputs
clustering/
    clustering.ipynb       k-Means from scratch, with outputs
predictions/               test-set predictions of every model + Kaggle submission files
results/                   tables of every experiment (CSV), validation predictions
Project_info/              the Kaggle data – NOT in git, see Setup
```

Still to do: **Decision Tree** and **Random Forest** (from scratch). Suggested place: `trees/trees.ipynb`.

## Setup

1. Clone this repository.
2. Download the data from the Kaggle **Data** tab and unzip the five CSV files into a folder called `Project_info/` in the project root (next to `common.py`).
3. Install the packages (Python 3.10+):

   ```
   pip install numpy pandas matplotlib torch scikit-learn
   ```

4. Open a notebook and choose **Run All**. Each notebook starts by moving to the project root, so `common.py`, `Project_info/`, `results/` and `predictions/` are found.

Running times on a laptop CPU: neural network 30–45 min, k-NN 20–30 min, clustering 5–10 min.

## Rules for every model (so results can be compared)

Use the shared split and preprocessing from `common.py`:

```python
import common

X, y, X_test, test_ids = common.load_task("image")        # or "sentiment"
train_idx, val_idx = common.stratified_split(y)            # 80/20, seed 42

stats = common.fit_preprocessing(X[train_idx], "sqrt")     # statistics from the training rows only
X_train = common.apply_preprocessing(X[train_idx], "sqrt", stats)
X_val   = common.apply_preprocessing(X[val_idx], "sqrt", stats)
```

- Choose models and hyperparameters with the **validation** split, not with the Kaggle leaderboard.
- Save test predictions as `predictions/<model>_<task>_test.csv` with columns `id,label`.
- `common.make_submission(...)` combines one image file and one sentiment file into a Kaggle submission.
- Required algorithms (k-NN, Decision Tree, Random Forest) must be written by us: no `sklearn` classifiers. `sklearn` is only used for metrics.

## Results so far (validation accuracy, same split for all)

| Method | Image | Sentiment |
|---|---|---|
| k-NN (first run: cosine) | 0.563 | 0.586 |
| k-NN (new: chi-squared + kernel voting, quick tests) | ~0.60 | ~0.60 |
| Decision Tree | – | – |
| Random Forest | – | – |
| Neural Network (3 seeds) | 0.432 | 0.679 |

Best per task so far: **k-NN for images, neural network for sentiment**.

Clustering (k-Means): ARI ≈ 0.02 on both tasks, so the classes are not natural clusters in either feature space.

## Kaggle files in `predictions/`

| File | Image model | Sentiment model | Expected score |
|---|---|---|---|
| `submission_knn_image_nn_sentiment.csv` | k-NN (first run) | Neural network | ≈ 0.62 |
| `submission_knn.csv` | k-NN | k-NN | ≈ 0.57 |
| `submission_nn.csv` | Neural network | Neural network | ≈ 0.56 |

Re-running `knn/knn.ipynb` rewrites the k-NN files with the chi-squared version (expected ≈ 0.64 combined with the NN for sentiment).

## Working together

- Pull before you start, push when you finish.
- One person per notebook: two people editing the same notebook at the same time is very hard to merge.
- Tell the team before changing `common.py`.

## External resources and assistance

- Claude (Anthropic) was used as a coding and analysis assistant for `common.py` and the notebooks.
- R. L. Rocha, "Bag of Visual Words applied to image classification", Medium – background for treating the image features as histograms (chi-squared distance).
