import os
import argparse
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from preprocess import load_metadata, join_with_image_paths, stratified_split, load_and_preprocess_image
from models import build_model
from tensorflow.keras.callbacks import ModelCheckpoint

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to metadata CSV")
    parser.add_argument("--img_dir", required=True, help="Directory with images")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    print("📥 Loading metadata...")
    df = load_metadata(args.csv)
    df = join_with_image_paths(df, args.img_dir)

    classes = sorted(df["dx"].unique())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    df["label"] = df["dx"].map(class_to_idx)

    print(f"✅ Found {len(classes)} classes: {classes}")

    train_df, val_df, test_df = stratified_split(df)

    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        rescale=1./255
    )

    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_dataframe(
        train_df,
        x_col="image_path",
        y_col="dx",
        target_size=(args.img_size, args.img_size),
        class_mode="categorical",
        batch_size=args.batch_size
    )

    val_gen = val_datagen.flow_from_dataframe(
        val_df,
        x_col="image_path",
        y_col="dx",
        target_size=(args.img_size, args.img_size),
        class_mode="categorical",
        batch_size=args.batch_size
    )

    model = build_model(num_classes=len(classes), input_shape=(args.img_size, args.img_size, 3))

    os.makedirs("models", exist_ok=True)
    checkpoint = ModelCheckpoint("models/skin_disease_model.h5", save_best_only=True, monitor="val_accuracy", mode="max")

    print("🚀 Starting training...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=args.epochs,
        callbacks=[checkpoint]
    )

    print("✅ Training complete! Model saved at 'models/skin_disease_model.h5'.")

if __name__ == "__main__":
    main()
