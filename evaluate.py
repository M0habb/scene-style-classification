import os
import torch
from models.vgg import VGG16
from utils.dataset import get_loaders
from utils.metrics import compute_all_metrics, plot_confusion_matrix, plot_all_models_comparison
from config import DEVICE, MODEL_SAVE_PATH


def evaluate(model_class, model_name, num_classes, class_names, save_dir="models"):
    """
    Loads the best saved checkpoint for a model and evaluates it
    on the test set. Produces metrics and confusion matrix.

    Args:
        model_class:  the model CLASS itself (not an instance)
        model_name:   string matching what was used in train()
        num_classes:  number of output classes
        class_names:  list of string class names
        save_dir:     where checkpoints and plots are saved
    Returns:
        dict of metrics: accuracy, precision, recall, f1
    """

    # ── 1. Reconstruct model and load saved weights ──────────────────
    model      = model_class(num_classes=num_classes).to(DEVICE)
    checkpoint = os.path.join(save_dir, f"{model_name}_best.pth")

    if not os.path.exists(checkpoint):
        print(f"  ⚠️  No checkpoint found for {model_name} at {checkpoint}")
        print(f"      Has this model been trained yet?")
        return None

    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()
    print(f"\n✅ Loaded weights from {checkpoint}")

    # ── 2. Get test loader ───────────────────────────────────────────
    _, _, test_loader, _, _ = get_loaders()

    # ── 3. Run inference on entire test set ──────────────────────────
    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)

            all_preds.extend( predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ── 4. Compute and print all metrics ─────────────────────────────
    metrics = compute_all_metrics(
        all_labels, all_preds,
        class_names, model_name
    )

    # ── 5. Plot confusion matrix ──────────────────────────────────────
    plot_confusion_matrix(
        all_labels, all_preds,
        class_names, model_name,
        save_dir=save_dir
    )

    return metrics


def evaluate_all(model_registry, num_classes, class_names, save_dir="models"):
    """
    Evaluates all trained models in sequence and prints a final
    comparison table with a grouped bar chart.

    Args:
        model_registry: dict of {model_name: model_class}
                        e.g. {"vgg": VGG16, "cnn": CustomCNN}
        num_classes:    number of output classes
        class_names:    list of string class names
        save_dir:       where checkpoints and plots are saved
    """
    all_results = {}

    for model_name, model_class in model_registry.items():
        print(f"\n{'='*60}")
        print(f"  Evaluating: {model_name.upper()}")
        print(f"{'='*60}")

        metrics = evaluate(
            model_class=model_class,
            model_name=model_name,
            num_classes=num_classes,
            class_names=class_names,
            save_dir=save_dir
        )

        if metrics is not None:
            all_results[model_name] = metrics

    # ── Final comparison table ────────────────────────────────────────
    if all_results:
        print("\n" + "="*60)
        print("  FINAL MODEL COMPARISON")
        print("="*60)
        print(f"  {'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-"*60)

        for name, m in all_results.items():
            print(f"  {name.upper():<20} "
                  f"{m['accuracy']:>10.4f} "
                  f"{m['precision']:>10.4f} "
                  f"{m['recall']:>10.4f} "
                  f"{m['f1']:>10.4f}")

        print("="*60)

        # Bar chart comparing all models
        plot_all_models_comparison(all_results, save_dir=save_dir)

    return all_results