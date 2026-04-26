from torchvision import transforms
from config import IMG_SIZE

# Training: augmentation to prevent overfitting
train_transforms = transforms.Compose([

    # Ensures all images same size
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    # Next 3 lines concerned with augmentation: Changing features in the images to make AI adapt to more features
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),

    # Convert images to tensors
    transforms.ToTensor(),

    # Concerned with gradient descent converging speed, skip for now
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])

# Validation/Test: NO augmentation, only deterministic preprocessing
val_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    ),
])