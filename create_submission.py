import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader

# 1. Custom Dataset for flat test folder (no class subfolders)
class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir  = test_dir
        self.transform = transform
        all_files = sorted([
            f for f in os.listdir(test_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        # Filter out corrupted/unreadable images upfront
        self.filenames = []
        skipped = 0
        for f in all_files:
            try:
                img = Image.open(os.path.join(test_dir, f))
                img.verify()  # checks file integrity without fully decoding
                self.filenames.append(f)
            except Exception:
                print(f"⚠️  Skipping corrupted image: {f}")
                skipped += 1
        
        print(f"✅ {len(self.filenames)} valid images | ⚠️  {skipped} skipped")

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img   = Image.open(os.path.join(self.test_dir, fname)).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, fname


# 2. Load class names from training data
_, _, class_names, num_classes = get_loaders()

# 3. Load best saved model
model = VGG16_V2(num_classes=num_classes).to(DEVICE)
model.load_state_dict(torch.load(
    "/kaggle/working/models/vgg_v2_best.pth",
    map_location=DEVICE
))
model.eval()

# 4. Build test loader
test_dataset = TestDataset(TEST_DIR, transform=val_transforms)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=NUM_WORKERS)

print(f"Found {len(test_dataset)} test images")

# 5. Run model
filenames_all = []
predictions   = []

with torch.no_grad():
    for images, fnames in test_loader:
        images  = images.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs, dim=1)

        filenames_all.extend(fnames)
        predictions.extend([class_names[p.item()] for p in predicted])

# 6. Build and save submission CSV
# class_names is alphabetically sorted by ImageFolder, so index = numeric class label
label_to_idx = {name: idx for idx, name in enumerate(class_names)}

submission = pd.DataFrame({
    "ImageName":  filenames_all,                              # keep full filename e.g. testimage_1.jpg
    "ClassLabel": [label_to_idx[p] for p in predictions]     # numeric: 0-16
})

SUBMISSION_PATH = "/kaggle/working/submission.csv"
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"✅ Submission saved → {SUBMISSION_PATH}")
print(f"   Total predictions: {len(submission)}")
print(submission.head(10))