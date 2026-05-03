import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

def accuracy(outputs, labels):
    """
    Computes the fraction of correctly classified samples in a batch.
    
    Args:
        outputs: tensor of shape (batch_size, num_classes) — raw model output
        labels:  tensor of shape (batch_size,) — ground truth class indices
    Returns:
        float between 0.0 and 1.0
    """
    _, predicted = torch.max(outputs, dim=1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)