import os
import shutil
from sklearn.model_selection import train_test_split

SOURCE_DIR = "data/train"
VAL_DIR = "data/val"

SPLIT_RATIO = 0.2  # 20% validation

def split_data():
    for class_name in os.listdir(SOURCE_DIR):
        class_path = os.path.join(SOURCE_DIR, class_name)

        if not os.path.isdir(class_path):
            continue

        images = os.listdir(class_path)

        train_imgs, val_imgs = train_test_split(
            images,
            test_size=SPLIT_RATIO,
            random_state=42,
            shuffle=True
        )

        # Create val class folder
        val_class_path = os.path.join(VAL_DIR, class_name)
        os.makedirs(val_class_path, exist_ok=True)

        # Move validation images
        for img in val_imgs:
            src = os.path.join(class_path, img)
            dst = os.path.join(val_class_path, img)
            shutil.move(src, dst)

    print("✅ Data split completed!")


if __name__ == "__main__":
    split_data()