# model_utils.py
# Shared utilities for the HAM10000 modelling pipeline.
# Import in your notebook with:
#
#   from model_utils import get_callbacks, plot_history, evaluate_model
#
# Place this file in the same directory as your notebook.

import os
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import keras
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from utils.utils_preproc import *



# ── Dataset ─────────────────────────────────────────────────────────────────
# VERSOES ANTIGAS
def load_image_with_resize(path, label, output_shape=(224, 224), preprocess_fn=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, output_shape)
    if preprocess_fn is not None:
        img = preprocess_fn(img)
    return img, label

def make_dataset(df, output_shape=(224, 224), shuffle=False, repeat=False, batch_size=32):
    paths  = df["image_path"].values
    labels = df["dx_encoded"].values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(
        lambda x, y: load_image_with_resize(x, y, output_shape),
        num_parallel_calls=2
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(1000, len(df)))
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size).prefetch(1)
    return ds


# VERSOES NOVAS
def load_image_new(path, label, resize_function=format_center_crop_tf, preprocess_fn=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = resize_function(img)
    if preprocess_fn is not None:
        img = preprocess_fn(img)
    return img, label

def make_dataset_new(df, shuffle=False, repeat=False, batch_size=32):
    paths  = df["image_path"].values
    labels = df["dx_encoded"].values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(
        lambda x, y: load_image_new(x, y),
        num_parallel_calls=2
    )
    if shuffle:
        ds = ds.shuffle(buffer_size=min(1000, len(df)))
    if repeat:
        ds = ds.repeat()
    ds = ds.batch(batch_size).prefetch(1)
    return ds

# ── Class Weights ─────────────────────────────────────────────────────────────────
def make_class_weights(df):
    classes = np.array(sorted(df["dx"].unique()))
    class_weights = compute_class_weight(class_weight="balanced",classes=classes,y=df["dx"])
    return {i: w for i, w in enumerate(class_weights)}


# ── Callbacks ─────────────────────────────────────────────────────────────────

def get_callbacks(checkpoint_path, patience_es=8, patience_lr=4):
    """
    Returns the standard callback list used by all models.

    Parameters
    ----------
    checkpoint_path : str
        Path where the best weights will be saved, e.g.
        "checkpoints/model_a_best.weights.h5"
    patience_es : int
        Number of epochs with no val_loss improvement before
        EarlyStopping halts training and restores the best weights.
    patience_lr : int
        Number of epochs with no val_loss improvement before
        ReduceLROnPlateau halves the learning rate.

    Returns
    -------
    list[keras.callbacks.Callback]
    """
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    return [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

class BatchTimeCallback(tf.keras.callbacks.Callback):
    """
    Custom Keras callback that records the duration of every
    training batch and epoch.

    After training, the recorded times can be used to compare computational
    cost across different augmentation strategies (offline vs. online).

    Attributes:
        batch_times (list[float]): Duration of each training batch in seconds.
        epoch_times (list[float]): Duration of each epoch in seconds.
    """

    def on_train_begin(self, logs=None):
        """Initialize empty lists to store batch and epoch durations."""
        self.batch_times = []
        self.epoch_times = []

    def on_epoch_begin(self, epoch, logs=None):
        """Record the start timestamp of the current epoch."""
        self._epoch_start = time.time()

    def on_train_batch_begin(self, batch, logs=None):
        """Record the start timestamp of the current batch."""
        self._batch_start = time.time()

    def on_train_batch_end(self, batch, logs=None):
        """Compute and store the elapsed time for the completed batch."""
        self.batch_times.append(time.time() - self._batch_start)

    def on_epoch_end(self, epoch, logs=None):
        """Compute and store the elapsed time for the completed epoch."""
        self.epoch_times.append(time.time() - self._epoch_start)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_history(history, title):
    """
    Plots loss and accuracy curves for a single Keras training History object.

    Parameters
    ----------
    history : keras.callbacks.History
        Object returned by model.fit().
    title : str
        Figure title, e.g. "Model B — EfficientNetB0 Phase 2".
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)

    axes[0].plot(history.history["loss"],         label="train")
    axes[0].plot(history.history["val_loss"],     label="val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history.history["accuracy"],     label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    plt.tight_layout()
    plt.show()


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(model, test_ds, label2idx, model_name="model"):
    """
    Full evaluation suite for a trained Keras model.

    Outputs
    -------
    - Classification report (precision, recall, F1 per class)
    - Melanoma recall highlighted separately (clinical priority metric)
    - Confusion matrix heatmap
    - Macro AUC-ROC score
    - Per-class ROC curves (melanoma plotted with a thicker line)

    Parameters
    ----------
    model : keras.Model
        Trained model. Assumed to output raw logits (from_logits=True).
    test_ds : tf.data.Dataset
        Unbatched or batched test dataset yielding (images, labels).
    label2idx : dict
        Mapping from class name to integer index, e.g. {"mel": 0, "nv": 1, ...}
    model_name : str
        Label used in plot titles and printed headers.

    Returns
    -------
    dict with keys "macro_auc" and "mel_recall"
    """
    n_classes  = len(label2idx)
    idx2label  = {v: k for k, v in label2idx.items()}
    class_names = [idx2label[i] for i in range(n_classes)]

    # ── Collect predictions ───────────────────────────────────────────────────
    y_true, y_pred_proba = [], []

    for images, labels in test_ds:
        proba = tf.nn.softmax(model(images, training=False)).numpy()
        y_pred_proba.append(proba)
        y_true.extend(labels.numpy())

    y_pred_proba = np.vstack(y_pred_proba)
    y_true       = np.array(y_true)
    y_pred       = np.argmax(y_pred_proba, axis=1)

    # ── Classification report ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  {model_name} — classification report")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=class_names))

    mel_idx    = label2idx["mel"]
    mel_recall = (y_pred[y_true == mel_idx] == mel_idx).mean()
    print(f"  *** Melanoma recall: {mel_recall:.3f} ***  (clinical priority metric)")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.title(f"{model_name} — confusion matrix")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.show()

    # ── Macro AUC-ROC ─────────────────────────────────────────────────────────
    y_bin     = label_binarize(y_true, classes=list(range(n_classes)))
    macro_auc = roc_auc_score(
        y_bin, y_pred_proba, average="macro", multi_class="ovr"
    )
    print(f"\n  Macro AUC-ROC: {macro_auc:.4f}")

    # ── Per-class ROC curves ──────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 7))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
        auc_score   = roc_auc_score(y_bin[:, i], y_pred_proba[:, i])
        lw = 2.5 if cls == "mel" else 1.2
        ax.plot(fpr, tpr, lw=lw, label=f"{cls} (AUC={auc_score:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{model_name} — per-class ROC curves  (melanoma in bold)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    return {"macro_auc": macro_auc, "mel_recall": mel_recall}


# ── Final comparison plot ──────────────────────────────────────────────────────

def plot_comparison(results: dict):
    """
    Bar chart comparing all trained models side by side.

    Parameters
    ----------
    results : dict
        Keys are model names, values are dicts with "macro_auc" and
        "mel_recall". Example:
            {
                "A — Custom CNN":      {"macro_auc": 0.91, "mel_recall": 0.78},
                "B — EfficientNetB0":  {"macro_auc": 0.96, "mel_recall": 0.85},
                "C — MobileNetV2":     {"macro_auc": 0.94, "mel_recall": 0.82},
            }
    """


    df = pd.DataFrame(results).T.reset_index()
    df.columns = ["Model", "Macro AUC", "Mel recall"]

    colors = ["#7F77DD", "#1D9E75", "#D85A30"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Model comparison — test set")

    axes[0].bar(df["Model"], df["Macro AUC"], color=colors[: len(df)])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Macro AUC-ROC")
    axes[0].tick_params(axis="x", rotation=15)

    axes[1].bar(df["Model"], df["Mel recall"], color=colors[: len(df)])
    axes[1].axhline(0.8, color="red", linestyle="--", lw=1, label="0.80 target")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Melanoma recall  (clinical priority)")
    axes[1].legend()
    axes[1].tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.show()

    print(df.to_string(index=False))