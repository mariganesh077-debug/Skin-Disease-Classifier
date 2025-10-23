import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

def load_metadata(csv_path):
    """Load dataset metadata CSV. Expect columns: 'image_id' and 'dx'."""
    df = pd.read_csv(csv_path)

    # Normalize column names
    df.columns = [c.strip().lower() for c in df.columns]
    if 'image_id' not in df.columns and 'imageid' in df.columns:
        df = df.rename(columns={'imageid': 'image_id'})
    if 'dx' not in df.columns:
        raise ValueError("metadata.csv must contain a 'dx' column (diagnosis labels).")

    return df


def join_with_image_paths(df, image_dir, image_exts=['.jpg', '.jpeg', '.png']):
    """Add a column 'image_path' containing full paths to each image."""
    paths = []
    for img_id in df['image_id']:
        found = None
        for ext in image_exts:
            candidate = os.path.join(image_dir, f"{img_id}{ext}")
            if os.path.exists(candidate):
                found = candidate
                break
        if found is None:
            candidate = os.path.join(image_dir, img_id)
            if os.path.exists(candidate):
                found = candidate
        paths.append(found)

    df = df.copy()
    df['image_path'] = paths
    missing = df['image_path'].isna().sum()
    if missing > 0:
        print(f"⚠️ Warning: {missing} images not found in {image_dir}.")
    return df


def stratified_split(df, label_col='dx', test_size=0.2, val_size=0.1, random_state=42):
    """Return train/val/test stratified splits."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df[label_col], random_state=random_state
    )
    val_fraction = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_df, test_size=val_fraction, stratify=train_df[label_col], random_state=random_state
    )
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def load_and_preprocess_image(img_path, img_size=224):
    """Load an image from path and resize it."""
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    img = img.astype("float32") / 255.0
    return img
