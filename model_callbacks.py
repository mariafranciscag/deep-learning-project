import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras
import tensorflow as tf

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


os.makedirs("checkpoints", exist_ok=True)

def get_callbacks(checkpoint_path, patience_es=8, patience_lr=4):
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience_es,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1
        ),
    ]

# %%
# ── SHARED PLOTTING ───────────────────────────────────────────────────────────
def plot_history(history, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(title)
    axes[0].plot(history.history["loss"],         label="train")
    axes[0].plot(history.history["val_loss"],     label="val")
    axes[0].set_title("Loss"); axes[0].legend()
    axes[1].plot(history.history["accuracy"],     label="train")
    axes[1].plot(history.history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy"); axes[1].legend()
    plt.tight_layout(); plt.show()

# %%
# ── SHARED EVALUATION ─────────────────────────────────────────────────────────
def evaluate_model(model, test_ds, test_df, label2idx, model_name="model"):
    idx2label    = {v: k for k, v in label2idx.items()}
    class_names  = [idx2label[i] for i in range(len(label2idx))]

    y_true, y_pred_proba = [], []
    for images, labels in test_ds:
        proba = tf.nn.softmax(model(images, training=False)).numpy()
        y_pred_proba.append(proba)
        y_true.extend(labels.numpy())

    y_pred_proba = np.vstack(y_pred_proba)
    y_true       = np.array(y_true)
    y_pred       = np.argmax(y_pred_proba, axis=1)

    print(f"\n{'='*60}\n  {model_name} — classification report\n{'='*60}")
    print(classification_report(y_true, y_pred, target_names=class_names))

    mel_idx    = label2idx["mel"]
    mel_recall = (y_pred[y_true == mel_idx] == mel_idx).mean()
    print(f"  *** Melanoma recall: {mel_recall:.3f} ***  (clinical priority metric)")

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"{model_name} — confusion matrix")
    plt.ylabel("True label"); plt.xlabel("Predicted label")
    plt.tight_layout(); plt.show()

    # AUC-ROC
    y_bin     = label_binarize(y_true, classes=list(range(N_CLASSES)))
    macro_auc = roc_auc_score(y_bin, y_pred_proba, average="macro", multi_class="ovr")
    print(f"\n  Macro AUC-ROC: {macro_auc:.4f}")

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, cls in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_proba[:, i])
        auc_score   = roc_auc_score(y_bin[:, i], y_pred_proba[:, i])
        lw = 2.5 if cls == "mel" else 1.2
        ax.plot(fpr, tpr, lw=lw, label=f"{cls} (AUC={auc_score:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title(f"{model_name} — per-class ROC curves  (melanoma in bold)")
    ax.legend(loc="lower right")
    plt.tight_layout(); plt.show()

    return {"macro_auc": macro_auc, "mel_recall": mel_recall}