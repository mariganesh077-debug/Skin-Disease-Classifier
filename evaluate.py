import argparse
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from preprocess import load_metadata, join_with_image_paths

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--img_dir", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    print("📥 Loading data...")
    df = load_metadata(args.csv)
    df = join_with_image_paths(df, args.img_dir)

    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_dataframe(
        df,
        x_col="image_path",
        y_col="dx",
        target_size=(args.img_size, args.img_size),
        class_mode="categorical",
        shuffle=False
    )

    print("🔍 Loading model...")
    model = load_model(args.model)

    print("🧠 Evaluating...")
    preds = model.predict(generator)
    y_pred = np.argmax(preds, axis=1)
    y_true = generator.classes

    print("\n📊 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=list(generator.class_indices.keys())))

    print("\n📉 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

if __name__ == "__main__":
    main()
