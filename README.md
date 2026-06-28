# Scene Style Classification

A PyTorch-based image classification framework that benchmarks multiple deep learning architectures — CNN, VGG16, ResNet, and Vision Transformer (ViT) — on a 17-class scene/style dataset. The project also includes a knowledge distillation pipeline that transfers learned representations from a ResNet teacher into a lighter ViT student.

---

## Features

- **Multi-model training** — swap between a custom CNN, VGG16, ResNet, and ViT from a single entry point (`train_model.py`)
- **Knowledge distillation** — transfer soft-label knowledge from a trained ResNet teacher to a ViT student (`train_distill.py`)
- **Centralised config** — all hyperparameters (batch size, epochs, learning rates, image size, class count) live in `config.py`; no hard-coding needed
- **Automatic checkpointing** — best validation-accuracy weights are saved per model to `models/`
- **Training curves** — loss and accuracy plots are generated and saved automatically after each run
- **Pretrained weight support** — commented-out scaffolding in `train_model.py` for loading and fine-tuning pretrained VGG16 weights
- **Submission generation** — `create_submission.py` produces competition-ready prediction files
- **GPU / CPU agnostic** — automatically uses CUDA if available, falls back to CPU

---

## Project Structure

```
scene-style-classification/
├── config.py              # All hyperparameters and paths
├── train.py               # Core training loop (loss, optimizer, scheduler, checkpointing)
├── train_model.py         # Entry point — instantiate and train a chosen model
├── train_distill.py       # Knowledge distillation: ResNet teacher → ViT student
├── evaluate.py            # Evaluation utilities (multi-model comparison)
├── predict.py             # Run inference on new images
├── create_submission.py   # Generate a predictions CSV for submission
├── models/
│   ├── cnn.py             # Custom CNN
│   ├── vgg.py             # VGG16 implementation
│   ├── resnet.py          # ResNet implementation
│   └── ViT.py             # Vision Transformer (ViT)
└── utils/
    ├── dataset.py         # DataLoader setup (train / val / test splits)
    └── metrics.py         # Accuracy and other metric helpers
```

---

## Setup

**Requirements:** Python 3.8+, PyTorch, torchvision, matplotlib

```bash
git clone https://github.com/M0habb/scene-style-classification.git
cd scene-style-classification
pip install torch torchvision matplotlib
```

---

## Data

Organise your dataset in the following directory layout (standard `ImageFolder` format):

```
data/
├── train/
│   ├── class_1/
│   ├── class_2/
│   └── ...
├── val/
│   └── ...
└── test/
    └── ...
```

The number of classes is inferred automatically from the folder structure. The default config expects **17 classes** and resizes all images to **224×224**.

---

## Configuration

Edit `config.py` to adjust training behaviour:

| Parameter | Default | Description |
|---|---|---|
| `BATCH_SIZE` | 32 | Samples per batch |
| `NUM_EPOCHS` | 20 | Training epochs |
| `LEARNING_RATE` | 1e-3 | LR for CNN / VGG / ResNet |
| `VIT_LEARNING_RATE` | 1e-4 | LR for ViT (transformers need a smaller LR) |
| `IMG_SIZE` | 224 | Image resize dimension |
| `NUM_CLASSES` | 17 | Set dynamically, but used as default |
| `MODEL_SAVE_PATH` | `models/best_model.pth` | Base path for checkpoints |

---

## Training

Open `train_model.py`, uncomment the model you want to train, and run:

```bash
python train_model.py
```

Each model gets its own checkpoint file (e.g. `models/vit_best.pth`) so models never overwrite each other. Training curves are saved to the `models/` directory.

### Available models

| Model | Notes |
|---|---|
| `cnn` | Lightweight custom CNN — good baseline |
| `VGG16` | Deep CNN; supports loading pretrained weights |
| `ResNet` | Residual network; used as distillation teacher |
| `ViT` | Vision Transformer; use `VIT_LEARNING_RATE` and enable `use_grad_clip=True` |

---

## Knowledge Distillation

Train a ResNet teacher first, then run distillation to produce a compact ViT student:

```python
from train_distill import train_distill

train_distill(teacher_path="models/resnet_best.pth")
```

The distillation loop uses a **temperature of 4.0** and **alpha of 0.7**, meaning 70% of the loss comes from soft teacher targets and 30% from ground-truth labels. The student is trained with cosine annealing and gradient clipping for stable transformer training. The best student checkpoint is saved to `models/vit_distilled_best.pth`.

---

## Inference

```bash
python predict.py
```

To generate a submission CSV:

```bash
python create_submission.py
```

---

## Design Notes

- **Gradient clipping** is applied automatically when `use_grad_clip=True` (recommended for ViT) to prevent exploding gradients.
- **StepLR** scheduler (decay ×0.1 every 7 epochs) is used for CNN/VGG/ResNet; **CosineAnnealingLR** is used for the distilled ViT.
- The Adam optimiser filters frozen parameters via `filter(lambda p: p.requires_grad, ...)`, which is essential when fine-tuning pretrained weights with some layers frozen.

---

## License

This project is open source. Feel free to use, modify, and build on it.
