"""
common.py -- code shared by every model in our COMP472 project.

Why this file exists:
    All four classifiers (k-NN, Decision Tree, Random Forest, Neural Network)
    must be compared fairly. So they all load the data the same way, use the
    SAME train/validation split, and are measured with the same metrics.

Functions in this file:
    set_seed(seed)                         make random results repeatable
    load_task(task)                        read the CSV files
    stratified_split(y)                    80% train / 20% validation
    fit_preprocessing(X_train, mode)       learn mean/std from training data
    apply_preprocessing(X, mode, stats)    transform any data with those stats
    accuracy, confusion_matrix, ...        metrics
    make_submission(...)                   write the Kaggle CSV

task is "image" (50 classes, 500 features) or "sentiment" (3 classes, 768 features).
"""

import os
import random

import numpy as np
import pandas as pd


# The folder this file is in, and the data folder inside it.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_DIR, "Project_info")

SEED = 42
N_CLASSES = {"image": 50, "sentiment": 3}
SENTIMENT_NAMES = ["Negative", "Neutral", "Positive"]


# ==========================================================================
# 1. Random seed
# ==========================================================================
def set_seed(seed=SEED):
    """Fix the random generators so that running the code twice gives the
    same results (the project asks for reproducible experiments)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass  # torch is only needed for the neural network


# ==========================================================================
# 2. Loading the data
# ==========================================================================
def load_task(task):
    """Read <task>_train.csv and <task>_test.csv.

    Returns:
        X_train   numbers, one row per sample, one column per feature
        y_train   the correct class of each training row
        X_test    test features (no labels)
        test_ids  the id of each test row, needed for the submission file
    """
    train_file = os.path.join(DATA_DIR, task + "_train.csv")
    test_file = os.path.join(DATA_DIR, task + "_test.csv")
    train = pd.read_csv(train_file)
    test = pd.read_csv(test_file)

    # The feature columns are called feature_0, feature_1, ...
    feature_columns = []
    for column in train.columns:
        if column.startswith("feature_"):
            feature_columns.append(column)

    X_train = train[feature_columns].to_numpy(dtype=np.float32)
    y_train = train["label"].to_numpy(dtype=np.int64)
    X_test = test[feature_columns].to_numpy(dtype=np.float32)
    test_ids = test["id"].to_numpy()
    return X_train, y_train, X_test, test_ids


# ==========================================================================
# 3. Train / validation split
# ==========================================================================
def stratified_split(y, val_fraction=0.2, seed=SEED):
    """Split the training rows into 80% train and 20% validation.

    "Stratified" means we split each class separately, so every class keeps
    the same proportions. Example for the image task: each class has 400
    rows -> 320 go to train and 80 go to validation.

    Returns two arrays of row numbers: train_idx and val_idx.
    """
    rng = np.random.default_rng(seed)
    train_idx = []
    val_idx = []

    for c in np.unique(y):
        rows_of_class = np.where(y == c)[0]   # row numbers whose label is c
        rng.shuffle(rows_of_class)            # random order (same every run)

        n_val = int(round(len(rows_of_class) * val_fraction))
        val_idx.extend(rows_of_class[:n_val])     # first 20% -> validation
        train_idx.extend(rows_of_class[n_val:])   # the rest  -> train

    return np.sort(np.array(train_idx)), np.sort(np.array(val_idx))


# ==========================================================================
# 4. Preprocessing
# ==========================================================================
# mode can be:
#   "none"              keep the features as they are
#   "standardize"       for every feature: (x - mean) / std
#   "sqrt"              take sqrt(x), then make each row have length 1.
#                       Made for the image features: they are >= 0 and mostly
#                       zero (a histogram). sqrt makes a few big values less
#                       dominant compared with many small values.
#   "sqrt_standardize"  "sqrt" first, then "standardize"
#
# IMPORTANT: the mean and std are computed on the TRAINING rows only, then
# reused for validation and test. Using validation/test rows to compute them
# would leak information about data the model is not supposed to see.

PREPROCESS_MODES = ["none", "standardize", "sqrt", "sqrt_standardize"]


def sqrt_and_normalize(X):
    """sqrt of every value, then divide each row by its length (L2 norm)."""
    X = np.sqrt(np.maximum(X, 0))
    row_lengths = np.linalg.norm(X, axis=1, keepdims=True)
    return X / (row_lengths + 1e-12)   # + tiny number to avoid dividing by 0


def fit_preprocessing(X_train, mode):
    """Learn what the preprocessing needs from the training data.
    Returns a dictionary with the mean and std of every feature (or nothing)."""
    if mode not in PREPROCESS_MODES:
        raise ValueError("unknown preprocessing mode: " + str(mode))

    stats = {}
    if mode == "standardize" or mode == "sqrt_standardize":
        if mode == "sqrt_standardize":
            X_train = sqrt_and_normalize(X_train)
        stats["mean"] = X_train.mean(axis=0)          # one mean per feature
        stats["std"] = X_train.std(axis=0) + 1e-8     # one std per feature
    return stats


def apply_preprocessing(X, mode, stats):
    """Transform X using the statistics learned by fit_preprocessing."""
    if mode == "sqrt" or mode == "sqrt_standardize":
        X = sqrt_and_normalize(X)
    if mode == "standardize" or mode == "sqrt_standardize":
        X = (X - stats["mean"]) / stats["std"]
    return X.astype(np.float32)


# ==========================================================================
# 5. Metrics
# ==========================================================================
def accuracy(y_true, y_pred):
    """Fraction of predictions that are correct."""
    correct = 0
    for t, p in zip(y_true, y_pred):
        if t == p:
            correct += 1
    return correct / len(y_true)


def confusion_matrix(y_true, y_pred, n_classes):
    """cm[i][j] = how many samples of true class i were predicted as class j.
    The diagonal (i == j) counts the correct predictions."""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    return cm


def per_class_recall(cm):
    """For each class: correct predictions / number of samples of that class."""
    recalls = []
    for i in range(len(cm)):
        total = cm[i].sum()
        recalls.append(cm[i][i] / total if total > 0 else 0.0)
    return np.array(recalls)


def per_class_precision(cm):
    """For each class: correct predictions / number of times we predicted it."""
    precisions = []
    for j in range(len(cm)):
        total = cm[:, j].sum()
        precisions.append(cm[j][j] / total if total > 0 else 0.0)
    return np.array(precisions)


def most_confused_pairs(cm, top=10):
    """The (true class, predicted class, count) pairs with the most mistakes."""
    mistakes = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                mistakes.append((i, j, int(cm[i][j])))
    mistakes.sort(key=lambda m: m[2], reverse=True)   # biggest count first
    return mistakes[:top]


# ==========================================================================
# 6. Kaggle submission
# ==========================================================================
def make_submission(image_ids, image_pred, sentiment_ids, sentiment_pred, out_path):
    """Combine image and sentiment predictions into one CSV for Kaggle.

    The rows must follow the order of sample_submission.csv, so we match
    predictions to it by id (as shown on the competition page).
    """
    sample = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

    image_df = pd.DataFrame({"id": image_ids, "label": image_pred})
    sentiment_df = pd.DataFrame({"id": sentiment_ids, "label": sentiment_pred})
    predictions = pd.concat([image_df, sentiment_df], ignore_index=True)

    submission = sample[["id"]].merge(predictions, on="id", how="left",
                                      validate="one_to_one")

    # Safety checks before writing the file
    assert len(submission) == len(sample), "wrong number of rows"   # 10,000
    assert submission["label"].notna().all(), "some ids have no prediction"
    submission["label"] = submission["label"].astype(int)
    is_image = submission["id"].str.startswith("img_")
    is_text = submission["id"].str.startswith("text_")
    assert submission.loc[is_image, "label"].between(0, 49).all()
    assert submission.loc[is_text, "label"].between(0, 2).all()

    submission.to_csv(out_path, index=False)
    return submission
