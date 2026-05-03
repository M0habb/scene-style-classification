import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from utils.dataset import get_loaders
from utils.metrics import accuracy
from config import DEVICE, NUM_EPOCHS, LEARNING_RATE, MODEL_SAVE_PATH

def train(model, model_name, lr=LEARNING_RATE, use_grad_clip=False):
    
    # Load Data
    train_loader, val_loader, class_names, num_classes = get_loaders()

    # Select Device for Model
    model = model.to(DEVICE)

    # Set Loss Function
    criterion = nn.CrossEntropyLoss()

    # Set Grad Descent
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    # Decays learning rate by factor of 0.1 every 7 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Unique save path per model so they don't overwrite each other
    save_dir  = os.path.dirname(MODEL_SAVE_PATH)
    save_path = os.path.join(save_dir, f"{model_name}_best.pth")
    os.makedirs(save_dir, exist_ok=True)

    # For Tracking
    best_val_acc = 0.0
    history = {
        "train_loss": [],
        "val_loss":   [],
        "train_acc":  [],
        "val_acc":    []
    }

    print(f"\n{'='*60}")
    print(f"  Training : {model_name.upper()}")
    print(f"  Device   : {DEVICE}")
    print(f"  Epochs   : {NUM_EPOCHS}")
    print(f"  LR       : {lr}")
    print(f"  Classes  : {num_classes}")
    print(f"{'='*60}\n")

    # Epoch Loop
    for epoch in range(NUM_EPOCHS):

        model.train()
        running_loss = 0.0
        running_acc  = 0.0

        for batch_idx, (images, labels) in enumerate(train_loader):

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Zero gradients from previous batch
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass computes gradients via backpropagation
            loss.backward()

            # Gradient clipping — recommended for ViT to prevent
            # exploding gradients during transformer training
            if use_grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Weight update
            optimizer.step()

            running_loss += loss.item()
            running_acc  += accuracy(outputs, labels)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_acc  = 0.0

        # No gradient computation needed during validation
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.to(DEVICE)

                outputs   = model(images)
                loss      = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc  += accuracy(outputs, labels)

        # Epoch Summary
        n_train = len(train_loader)
        n_val   = len(val_loader)

        epoch_train_loss = running_loss / n_train
        epoch_train_acc  = running_acc  / n_train
        epoch_val_loss   = val_loss     / n_val
        epoch_val_acc    = val_acc      / n_val

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_acc"].append(epoch_val_acc)

        print(f"Epoch [{epoch+1:02d}/{NUM_EPOCHS}] "
              f"Train Loss: {epoch_train_loss:.4f}  Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f}  Acc: {epoch_val_acc:.4f}")

        scheduler.step()

        # Save checkpoint if best validation accuracy so far
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Best model saved → {save_path}  (val_acc={best_val_acc:.4f})")

    # Post Training
    print(f"\n{'='*60}")
    print(f"  {model_name.upper()} Training Complete")
    print(f"  Best Val Accuracy : {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"{'='*60}\n")

    _plot_history(history, model_name, save_dir)

    return best_val_acc, save_path


def _plot_history(history, model_name, save_dir):
    """
    Plots and saves training/validation loss and accuracy curves.
    Prefixed with _ to indicate this is an internal helper function
    not intended to be called directly from outside this file.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{model_name.upper()} — Training Curves", fontsize=14, fontweight="bold")

    # Loss plot
    ax1.plot(history["train_loss"], label="Train Loss", linewidth=2)
    ax1.plot(history["val_loss"],   label="Val Loss",   linewidth=2, linestyle="--")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(history["train_acc"], label="Train Acc", linewidth=2)
    ax2.plot(history["val_acc"],   label="Val Acc",   linewidth=2, linestyle="--")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(save_dir, f"{model_name}_training_curves.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"  📊 Training curves saved → {plot_path}")
    plt.show()