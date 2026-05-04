# utils/metrics.py
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


def accuracy(outputs, labels):
    """
    Computes fraction of correctly classified samples in a batch.
    Used inside the training loop after every batch.

    Args:
        outputs: tensor (batch_size, num_classes) — raw model logits
        labels:  tensor (batch_size,) — ground truth class indices
    Returns:
        float between 0.0 and 1.0
    """
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)


def compute_all_metrics(all_labels, all_preds, class_names, model_name="Model"):
    """
    Computes full suite of metrics over an entire dataset split.
    Call this AFTER collecting all predictions — not inside a batch loop.

    Args:
        all_labels:  list/array of true class indices
        all_preds:   list/array of predicted class indices
        class_names: list of string class names
        model_name:  string used in printed header
    Returns:
        dict with keys: accuracy, precision, recall, f1
    """
    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    acc       = (all_labels == all_preds).mean()
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall    = recall_score(   all_labels, all_preds, average='macro', zero_division=0)
    f1        = f1_score(       all_labels, all_preds, average='macro', zero_division=0)

    print("\n" + "="*60)
    print(f"  {model_name.upper()} — Test Set Results")
    print("="*60)
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print("="*60)
    print("\nPer-Class Report:")
    print(classification_report(
        all_labels, all_preds,
        target_names=class_names,
        zero_division=0
    ))

    return {
        "accuracy":  acc,
        "precision": precision,
        "recall":    recall,
        "f1":        f1
    }


def plot_confusion_matrix(all_labels, all_preds, class_names,
                          model_name="Model", save_dir="models"):
    """
    Plots and saves both raw count and normalized confusion matrices.

    Rows    = true classes
    Columns = predicted classes
    Diagonal = correct predictions

    Args:
        all_labels:  list/array of true class indices
        all_preds:   list/array of predicted class indices
        class_names: list of string class names
        model_name:  string used in plot title and filename
        save_dir:    directory to save the plot
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)

    cm            = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle(f"{model_name.upper()} — Confusion Matrix", fontsize=15, fontweight="bold")

    # --- Left: raw counts ---
    im1 = axes[0].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[0].set_title("Raw Counts", fontsize=12)
    plt.colorbar(im1, ax=axes[0])

    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j, i, str(cm[i, j]),
                         ha='center', va='center', fontsize=7,
                         color='white' if cm[i, j] > thresh else 'black')

    axes[0].set_xticks(range(len(class_names)))
    axes[0].set_yticks(range(len(class_names)))
    axes[0].set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    axes[0].set_yticklabels(class_names, fontsize=8)
    axes[0].set_ylabel('True Label',      fontsize=11)
    axes[0].set_xlabel('Predicted Label', fontsize=11)

    # --- Right: normalized ---
    im2 = axes[1].imshow(cm_normalized, interpolation='nearest',
                         cmap=plt.cm.Greens, vmin=0, vmax=1)
    axes[1].set_title("Normalized (Row %)", fontsize=12)
    plt.colorbar(im2, ax=axes[1])

    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            axes[1].text(j, i, f"{cm_normalized[i, j]:.2f}",
                         ha='center', va='center', fontsize=7,
                         color='white' if cm_normalized[i, j] > 0.5 else 'black')

    axes[1].set_xticks(range(len(class_names)))
    axes[1].set_yticks(range(len(class_names)))
    axes[1].set_xticklabels(class_names, rotation=45, ha='right', fontsize=8)
    axes[1].set_yticklabels(class_names, fontsize=8)
    axes[1].set_ylabel('True Label',      fontsize=11)
    axes[1].set_xlabel('Predicted Label', fontsize=11)

    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{model_name}_confusion_matrix.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  📊 Confusion matrix saved → {save_path}")
    plt.show()


def plot_all_models_comparison(all_results, save_dir="models"):
    """
    Plots a grouped bar chart comparing all four models across
    accuracy, precision, recall and F1.

    Args:
        all_results: dict of {model_name: metrics_dict}
                     e.g. {"vgg": {"accuracy": 0.85, ...}, ...}
        save_dir:    directory to save the plot
    """
    import os
    os.makedirs(save_dir, exist_ok=True)

    model_names = list(all_results.keys())
    metrics     = ["accuracy", "precision", "recall", "f1"]
    x           = np.arange(len(metrics))
    width       = 0.8 / len(model_names)
    colors      = ['#2196F3', '#4CAF50', '#FF9800', '#9C27B0']

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, (name, results) in enumerate(all_results.items()):
        values = [results[m] for m in metrics]
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars   = ax.bar(x + offset, values, width,
                        label=name.upper(), color=colors[i % len(colors)], alpha=0.85)

        # Label each bar with its value
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{val:.2f}", ha='center', va='bottom', fontsize=8)

    ax.set_title("Model Comparison — All Metrics", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics], fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(save_dir, "all_models_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  📊 Comparison chart saved → {save_path}")
    plt.show()