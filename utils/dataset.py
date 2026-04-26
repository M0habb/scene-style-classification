from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from utils.transforms import train_transforms, val_transforms
from config import TRAIN_DIR, VAL_DIR, TEST_DIR, BATCH_SIZE, NUM_WORKERS

def get_loaders():
    train_dataset = ImageFolder(root=TRAIN_DIR, transform=train_transforms)
    val_dataset   = ImageFolder(root=VAL_DIR,   transform=val_transforms)
    test_dataset  = ImageFolder(root=TEST_DIR,  transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    class_names = train_dataset.classes   # asian/boho/coastal...
    num_classes = len(class_names) # 17

    return train_loader, val_loader, test_loader, class_names, num_classes