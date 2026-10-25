"""
CNN-LSTM for LOS/NLOS classification: shared code for cnn_lstm_los_nlos.ipynb.

Used two ways:
  * imported by the notebook (Part A: the paper's split, with plots);
  * run as a script, one process per split (Part B: all 80 splits). A fresh
    process frees all TensorFlow memory when it exits, so memory doesn't grow
    from split to split.

Fixes compared with the first full-resolution version (84.6% over 60 splits,
validation 99.7% but unseen environments ~85%):
  1. the original architecture: valid padding, Dense(32), L2 = 0.001, Dropout 0.4;
  2. compact input per frame: 52 subcarrier amplitudes (mean over the 64
     antennas of the per-antenna-normalised CSI) instead of 3328 values;
  3. the number of epochs is chosen on UNSEEN environments: inner folds hold
     out whole training clusters, and the final model is trained on all
     training clusters for the epoch count with the lowest mean inner
     validation loss.

Script usage (normally done by the notebook):
  python cnn_lstm_core.py --config results/part_b_config.json --split '{"name": ..., "train": [...], "test": [...]}'
Prints one line "RESULT {json}".
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import keras
from keras import layers, regularizers
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score, recall_score, roc_auc_score

DEFAULT_CONFIG = {
    "input": "subcarrier",       # "subcarrier" (52) | "pca" (n_pca components of 3328) | "full" (3328)
    "n_pca": 64,
    "conv_filters": [64, 128],
    "kernel_size": 3,
    "pool_size": 2,
    "padding": "valid",
    "lstm_units": [128, 64],
    "dense_units": 32,
    "dropout": 0.4,
    "l2": 1e-3,
    "noise_std": 0.0,            # > 0 adds a GaussianNoise layer at the input (training only)
    "learning_rate": 1e-3,
    "batch_size": 32,
    "max_epochs": 50,
    "inner_patience": 10,        # early stopping inside each inner fold (curves are recorded)
    "window": 50,
    "stride": 10,
    "seed": 42,
}


# ----------------------------------------------------------------------------- data

def load_cluster_frames(data_dir, clusters, input_mode):
    """Per-cluster frame features (before z-score) in the chosen input form."""
    cluster_dir = os.path.join(data_dir, "clusters")
    with open(os.path.join(cluster_dir, "index.json")) as f:
        cidx = json.load(f)
    assert cidx["norm"] == "antenna" and not cidx["zscore_applied"], "re-run preprocess_raw.ipynb with NORM = 'antenna'"
    frames, labels = {}, {}
    for c in clusters:
        F = np.load(os.path.join(cluster_dir, cidx["clusters"][c]["features_file"]), mmap_mode="r")
        if input_mode == "subcarrier":
            frames[c] = np.asarray(F, dtype=np.float32).reshape(len(F), 64, 52).mean(axis=1)
        else:                                            # "full" and "pca" start from all 3328 values
            frames[c] = np.asarray(F, dtype=np.float32)
        labels[c] = int(cidx["clusters"][c]["label"])
    return frames, labels


class Transform:
    """Input transform fitted on training clusters only: optional PCA, then z-score."""

    def __init__(self, cfg, frames, train_clusters):
        X = np.concatenate([frames[c] for c in train_clusters])
        self.pca = None
        if cfg["input"] == "pca":
            self.pca = PCA(n_components=cfg["n_pca"], random_state=cfg["seed"]).fit(X)
            X = self.pca.transform(X)
        self.mu = X.mean(axis=0).astype(np.float32)
        self.sd = (X.std(axis=0) + 1e-8).astype(np.float32)

    def __call__(self, X):
        if self.pca is not None:
            X = self.pca.transform(X)
        return ((X - self.mu) / self.sd).astype(np.float32)


def make_sequences(frames, labels, clusters, transform, window, stride):
    """Windows of `window` frames (stride `stride`) cut inside each cluster."""
    Xs, ys, gs = [], [], []
    for c in clusters:
        Z = transform(frames[c])
        n = (len(Z) - window) // stride + 1
        idx = stride * np.arange(n)[:, None] + np.arange(window)[None, :]
        Xs.append(Z[idx]); ys.append(np.full(n, labels[c], dtype=np.float32)); gs.append(np.full(n, c))
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(gs)


# ----------------------------------------------------------------------------- model

def build_model(cfg, window, n_features):
    """The paper's CNN-LSTM, with the original layer settings."""
    inputs = keras.Input(shape=(window, n_features))
    x = inputs
    if cfg["noise_std"] > 0:
        x = layers.GaussianNoise(cfg["noise_std"])(x)
    for filters in cfg["conv_filters"]:
        x = layers.Conv1D(filters, cfg["kernel_size"], padding=cfg["padding"], activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(cfg["pool_size"])(x)
    x = layers.LSTM(cfg["lstm_units"][0], return_sequences=True)(x)
    x = layers.LSTM(cfg["lstm_units"][1])(x)
    x = layers.Dense(cfg["dense_units"], activation="relu", kernel_regularizer=regularizers.l2(cfg["l2"]))(x)
    x = layers.Dropout(cfg["dropout"])(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs, name="cnn_lstm")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=cfg["learning_rate"]),
                  loss="binary_crossentropy", metrics=["accuracy"])
    return model


# ----------------------------------------------------------------------------- epoch selection

def inner_folds(train_clusters):
    """Hold out one LOS + one NLOS training cluster per fold; every training cluster is held out at least once."""
    los = [c for c in train_clusters if c.startswith("LOS")]
    nlos = [c for c in train_clusters if c.startswith("NLOS")]
    folds = []
    for i in range(max(len(los), len(nlos))):
        held = [los[i % len(los)], nlos[i % len(nlos)]]
        folds.append({"train": [c for c in train_clusters if c not in held], "val": held})
    return folds


def select_epochs(cfg, frames, labels, train_clusters, verbose=False):
    """Train on each inner fold, watch the loss on the held-out clusters, return the best epoch count."""
    W, S, E = cfg["window"], cfg["stride"], cfg["max_epochs"]
    curves = []
    for k, fold in enumerate(inner_folds(train_clusters)):
        keras.backend.clear_session()
        keras.utils.set_random_seed(cfg["seed"] + k)
        tf_ = Transform(cfg, frames, fold["train"])
        Xtr, ytr, _ = make_sequences(frames, labels, fold["train"], tf_, W, S)
        Xva, yva, _ = make_sequences(frames, labels, fold["val"], tf_, W, S)
        model = build_model(cfg, W, Xtr.shape[2])
        stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=cfg["inner_patience"])
        h = model.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=E, batch_size=cfg["batch_size"],
                      callbacks=[stop], verbose=0)
        vl, va = np.array(h.history["val_loss"]), np.array(h.history["val_accuracy"])
        # folds that stopped early keep their last value for the remaining epochs
        vl = np.concatenate([vl, np.full(E - len(vl), vl[-1])])
        va = np.concatenate([va, np.full(E - len(va), va[-1])])
        curves.append({"held_out": fold["val"], "val_loss": vl.tolist(), "val_accuracy": va.tolist(),
                       "epochs_run": len(h.history["val_loss"])})
        if verbose:
            print(f"  inner fold {k + 1}: held out {fold['val']}  best val_loss {vl.min():.4f} at epoch "
                  f"{int(vl.argmin()) + 1}, val_acc there {va[vl.argmin()]:.4f}  ({len(h.history['val_loss'])} epochs run)")
        del model, h, Xtr, Xva
    mean_loss = np.mean([c["val_loss"] for c in curves], axis=0)
    mean_acc = np.mean([c["val_accuracy"] for c in curves], axis=0)
    best = int(mean_loss.argmin()) + 1
    return best, {"curves": curves, "mean_val_loss": mean_loss.tolist(), "mean_val_accuracy": mean_acc.tolist(),
                  "best_epochs": best, "inner_cv_accuracy": float(mean_acc[best - 1])}


# ----------------------------------------------------------------------------- one split

def train_and_evaluate(cfg, frames, labels, split, verbose=False):
    """Epoch selection on inner folds, final training on all training clusters, evaluation on the test clusters."""
    W, S = cfg["window"], cfg["stride"]
    t0 = time.time()
    best_epochs, sel = select_epochs(cfg, frames, labels, split["train"], verbose=verbose)
    t_sel = time.time() - t0

    keras.backend.clear_session()
    keras.utils.set_random_seed(cfg["seed"])
    tf_ = Transform(cfg, frames, split["train"])
    Xtr, ytr, _ = make_sequences(frames, labels, split["train"], tf_, W, S)
    model = build_model(cfg, W, Xtr.shape[2])
    t0 = time.time()
    h = model.fit(Xtr, ytr, epochs=best_epochs, batch_size=cfg["batch_size"], verbose=0)
    t_fit = time.time() - t0
    train_acc = float(((model.predict(Xtr, batch_size=256, verbose=0).ravel() >= 0.5) == ytr).mean())

    Xte, yte, gte = make_sequences(frames, labels, split["test"], tf_, W, S)
    p = model.predict(Xte, batch_size=256, verbose=0).ravel()
    pred = (p >= 0.5).astype(np.int8)
    per_cluster = {c: float((pred[gte == c] == yte[gte == c]).mean()) for c in split["test"]}

    result = {
        "name": split["name"], "train": split["train"], "test": split["test"],
        "n_train_windows": int(len(ytr)), "n_test_windows": int(len(yte)),
        "accuracy": float((pred == yte).mean()),
        "macro_f1": float(f1_score(yte, pred, average="macro")),
        "roc_auc": float(roc_auc_score(yte, p)),
        "recall_los": float(recall_score(yte, pred, pos_label=1)),
        "recall_nlos": float(recall_score(yte, pred, pos_label=0)),
        "per_cluster_accuracy": per_cluster,
        "selected_epochs": best_epochs,
        "inner_cv_accuracy": sel["inner_cv_accuracy"],      # estimate for unseen environments, from training clusters only
        "train_accuracy": train_acc,
        "select_seconds": t_sel, "fit_seconds": t_fit,
    }
    details = {"model": model, "history": {k: [float(v) for v in vals] for k, vals in h.history.items()},
               "selection": sel, "p_los": p, "y_true": yte, "y_pred": pred, "groups": gte, "transform": tf_}
    return result, details


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--split", required=True, help="JSON with name, train and test cluster lists")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = json.load(f)
    split = json.loads(args.split)
    frames, labels = load_cluster_frames(cfg["data_dir"], split["train"] + split["test"], cfg["input"])
    result, _ = train_and_evaluate(cfg, frames, labels, split)
    sys.stdout.write("RESULT " + json.dumps(result) + "\n")


if __name__ == "__main__":
    main()
