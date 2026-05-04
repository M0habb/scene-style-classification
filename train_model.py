from models.vgg import VGG16
from utils.dataset import get_loaders
from train import train
import torch
from models import cnn
from config import DEVICE, LEARNING_RATE, VIT_LEARNING_RATE
import gc
from evaluate import evaluate_all

# READ: This file is for training the models. If your model is not pretrained, comment out entire pretrained
# section. Leave get_loaders() line as is. Instantiate your model (replace with ur model class and rename)
# Change the train function call at the bottom (replace parameters with ur model instance and name)
# Tweak paramaters (batch size, epochs, learning rate) from CONFIG. DO NOT HARD CODE (except for model exceptions)


# Clear previous model from GPU memory
gc.collect()
torch.cuda.empty_cache()

# Get num_classes from loaders
_, _, class_names, num_classes = get_loaders()

# Instantiate model
#vgg_model = VGG16(num_classes=num_classes)
cnn_model = cnn()

# LOADING PRETRAINED WEIGHTS
# PRETRAINED_PATH = "pretrained/vgg16_bn.pth"

# # Load the pretrained state dict
# pretrained_dict = torch.load(PRETRAINED_PATH, map_location=DEVICE)

# # Get the model's own state dict
# model_dict = vgg_model.state_dict()
# pretrained_dict = {
#     k: v for k, v in pretrained_dict.items()
#     if k in model_dict and model_dict[k].shape == v.shape
# }

# # Update model dict with pretrained weights
# model_dict.update(pretrained_dict)
# vgg_model.load_state_dict(model_dict)

# print(f"✅ Pretrained weights loaded — {len(pretrained_dict)} layers transferred")
# print(f"   Classifier layers kept random (shape mismatch with ImageNet 1000 classes)")


# Train
best_acc, save_path = train(
    model      = cnn_model,
    model_name = "cnn_v1",
    lr         = LEARNING_RATE,       # lower LR when using pretrained weights
)

# best_acc, save_path = train(
#     model      = vit_model,
#     model_name = "vit",
#     lr         = VIT_LEARNING_RATE,
#     use_grad_clip = True
# )

# Add each model to this dict as you finish training them
model_registry = {
    "vgg_pretrained": VGG16,
    # "vgg":    VGG16,
    "cnn":    cnn,
    # "resnet": CustomResNet,
    # "vit":    CustomViT,
}

#all_results = evaluate_all(
#    model_registry=model_registry,
#    num_classes=num_classes,
#    class_names=class_names,
#    save_dir="models"
#)