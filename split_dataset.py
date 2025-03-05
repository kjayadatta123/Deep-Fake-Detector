import os
import shutil
from sklearn.model_selection import train_test_split

source_dir = 'dataset'
output_dir = 'dataset_split'
categories = ['REAL', 'FAKE']
split_ratio = 0.8

for category in categories:
    category_path = os.path.join(source_dir, category)
    images = os.listdir(category_path)

    train_images, val_images = train_test_split(images, train_size=split_ratio, random_state=42)

    os.makedirs(os.path.join(output_dir, 'train', category), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', category), exist_ok=True)

    for img in train_images:
        shutil.copy(
            os.path.join(category_path, img),
            os.path.join(output_dir, 'train', category, img)
        )

    for img in val_images:
        shutil.copy(
            os.path.join(category_path, img),
            os.path.join(output_dir, 'val', category, img)
        )

print("✅ Dataset successfully split into train and validation sets!")
