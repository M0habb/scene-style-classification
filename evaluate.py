from models.vgg import VGG16
from utils.dataset import get_loaders
from train import train

# Get num_classes from loaders
_, _, class_names, num_classes = get_loaders()

# Instantiate model
vgg_model = VGG16(num_classes=num_classes)

# Train
best_acc, save_path = train(
    model      = vgg_model,
    model_name = "vgg",
    lr         = LEARNING_RATE,
)