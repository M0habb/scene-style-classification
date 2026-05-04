import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from utils.dataset import get_loaders
from utils.metrics import accuracy
from models.resnet import ResNet
from models.ViT import ViT
from config import DEVICE, NUM_EPOCHS, VIT_LEARNING_RATE


# ── Distillation hyperparameters ─────────────────────────────────────────────
TEMPERATURE = 4.0   # T > 1 softens the teacher's probability distribution,
                    # revealing more signal in the "dark knowledge" (near-zero probs)
                    # Higher T = softer targets = more inter-class info transferred

ALPHA = 0.7         # Weight on the soft (teacher) loss.
                    # 1 - ALPHA goes to hard (ground truth) loss.
                    # 0.7 means: "trust the teacher more than the labels"
# ─────────────────────────────────────────────────────────────────────────────


def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    Combines two losses:

      1. Soft loss  — KL divergence between softened teacher & student distributions
                      This is the "knowledge transfer" part.
                      Dividing logits by T before softmax makes the distribution
                      flatter, so the student learns from small probability differences.

      2. Hard loss  — Standard cross entropy against ground truth labels
                      Keeps the student grounded to the actual correct answer.

    Final loss = alpha * soft + (1 - alpha) * hard
    """

    # Soft targets: divide by T before softmax to "soften" the distribution
    soft_student  = F.log_softmax(student_logits / T, dim=1)
    soft_teacher  = F.softmax(teacher_logits    / T, dim=1)

    # KL divergence. Multiply by T^2 to rescale gradients back to normal scale
    # (dividing by T shrinks gradients by 1/T^2, so we compensate here)
    soft_loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean') * (T ** 2)

    # Hard targets: standard cross entropy
    hard_loss = F.cross_entropy(student_logits, labels)

    return alpha * soft_loss + (1 - alpha) * hard_loss


def train_distill(
    teacher_path: str,      # path to resnet_best.pth
    save_dir:     str = "models",
    lr:           float = VIT_LEARNING_RATE,
):
    os.makedirs(save_dir, exist_ok=True)

    train_loader, val_loader, class_names, num_classes = get_loaders()

    # ── 1. Load teacher (ResNet) — freeze it completely ──────────────
    teacher = ResNet(num_classes=num_classes).to(DEVICE)
    teacher.load_state_dict(torch.load(teacher_path, map_location=DEVICE))
    teacher.eval()
    for param in teacher.parameters():         # teacher never updates
        param.requires_grad = False
    print(f"✅ Teacher loaded from {teacher_path}")

    # ── 2. Build student (ViT) ────────────────────────────────────────
    student = ViT(num_classes=num_classes).to(DEVICE)

    # ── 3. Optimizer + scheduler (student only) ───────────────────────
    optimizer = optim.Adam(student.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    #   CosineAnnealing works better than StepLR for ViT — smoothly decays
    #   LR to near-zero rather than dropping it in sudden steps

    # ── 4. Tracking ───────────────────────────────────────────────────
    best_val_acc = 0.0
    save_path    = os.path.join(save_dir, "vit_distilled_best.pth")
    history      = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    print(f"\n{'='*60}")
    print(f"  Knowledge Distillation: ResNet → ViT")
    print(f"  Temperature : {TEMPERATURE}")
    print(f"  Alpha       : {ALPHA}  (soft) / {1-ALPHA:.1f} (hard)")
    print(f"  Device      : {DEVICE}")
    print(f"  Epochs      : {NUM_EPOCHS}")
    print(f"{'='*60}\n")

    for epoch in range(NUM_EPOCHS):

        # ── Train ─────────────────────────────────────────────────────
        student.train()
        # teacher stays in eval() the entire time — never call teacher.train()

        running_loss = 0.0
        running_acc  = 0.0

        for images, labels in train_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Get teacher logits — no gradient needed
            with torch.no_grad():
                teacher_logits = teacher(images)

            # Get student logits — gradient flows here
            student_logits = student(images)

            # Combined distillation loss
            loss = distillation_loss(
                student_logits, teacher_logits, labels,
                T=TEMPERATURE, alpha=ALPHA
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            # Grad clipping is important for ViT — transformers can have
            # unstable gradients especially early in training
            optimizer.step()

            running_loss += loss.item()
            running_acc  += accuracy(student_logits, labels)

        # ── Validate ──────────────────────────────────────────────────
        student.eval()
        val_loss = 0.0
        val_acc  = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                teacher_logits = teacher(images)
                student_logits = student(images)

                loss      = distillation_loss(
                    student_logits, teacher_logits, labels,
                    T=TEMPERATURE, alpha=ALPHA
                )
                val_loss += loss.item()
                val_acc  += accuracy(student_logits, labels)

        # ── Epoch summary ─────────────────────────────────────────────
        n_train = len(train_loader)
        n_val   = len(val_loader)

        e_train_loss = running_loss / n_train
        e_train_acc  = running_acc  / n_train
        e_val_loss   = val_loss     / n_val
        e_val_acc    = val_acc      / n_val

        history["train_loss"].append(e_train_loss)
        history["val_loss"].append(e_val_loss)
        history["train_acc"].append(e_train_acc)
        history["val_acc"].append(e_val_acc)

        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
              f"Train Loss: {e_train_loss:.4f}  Acc: {e_train_acc:.4f} | "
              f"Val Loss: {e_val_loss:.4f}  Acc: {e_val_acc:.4f}")

        scheduler.step()

        if e_val_acc > best_val_acc:
            best_val_acc = e_val_acc
            torch.save(student.state_dict(), save_path)
            print(f"  ✅ Best student saved → {save_path}  (val_acc={best_val_acc:.4f})")

    print(f"\n{'='*60}")
    print(f"  Distillation Complete")
    print(f"  Best Val Accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"{'='*60}\n")

    _plot_history(history, save_dir)
    return best_val_acc, save_path


def _plot_history(history, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ViT (Distilled from ResNet) — Training Curves",
                 fontsize=14, fontweight="bold")

    ax1.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(history["val_loss"],   label="Val Loss",   linewidth=2, linestyle="--")
    ax1.set_title("Loss");  ax1.set_xlabel("Epoch");  ax1.set_ylabel("Loss")
    ax1.legend();           ax1.grid(True, alpha=0.3)

    ax2.plot(history["train_acc"], label="Train Acc", linewidth=2)
    ax2.plot(history["val_acc"],   label="Val Acc",   linewidth=2, linestyle="--")
    ax2.set_title("Accuracy");  ax2.set_xlabel("Epoch");  ax2.set_ylabel("Accuracy")
    ax2.legend();               ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "vit_distilled_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  📊 Training curves saved → {path}")
    plt.show()