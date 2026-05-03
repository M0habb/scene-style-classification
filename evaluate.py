from models.vgg import VGG16
from utils.dataset import get_loaders
from train import train
import torch
from config import DEVICE
import gc

# Clear previous model from GPU memory
gc.collect()
torch.cuda.empty_cache()
# Get num_classes from loaders
_, _, class_names, num_classes = get_loaders()

# Instantiate model
vgg_model = VGG16(num_classes=num_classes)

# ── Load pretrained weights ──────────────────────────────────────────
PRETRAINED_PATH = "pretrained/vgg16_bn.pth"

# Load the pretrained state dict
pretrained_dict = torch.load(PRETRAINED_PATH, map_location=DEVICE)

# Get the model's own state dict
model_dict = vgg_model.state_dict()

# Filter out the classifier layers — their shapes won't match because
# the pretrained model has 1000 output classes (ImageNet) but yours
# has num_classes (17). We only load the features (conv) layers.
pretrained_dict = {
    k: v for k, v in pretrained_dict.items()
    if k in model_dict and model_dict[k].shape == v.shape
}

# Update model dict with pretrained weights
model_dict.update(pretrained_dict)
vgg_model.load_state_dict(model_dict)

print(f"✅ Pretrained weights loaded — {len(pretrained_dict)} layers transferred")
print(f"   Classifier layers kept random (shape mismatch with ImageNet 1000 classes)")
# ────────────────────────────────────────────────────────────────────

# Train
best_acc, save_path = train(
    model      = vgg_model,
    model_name = "vgg_pretrained",
    lr         = 1e-4,       # lower LR when using pretrained weights
)