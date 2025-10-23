import argparse
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from preprocess import load_and_preprocess_image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()

    print("🔍 Loading model...")
    model = load_model(args.model)

    print(f"📷 Loading image: {args.image}")
    img = load_and_preprocess_image(args.image, args.img_size)
    img_batch = np.expand_dims(img, axis=0)

    preds = model.predict(img_batch)[0]
    class_idx = np.argmax(preds)
    confidence = preds[class_idx]

    print(f"✅ Predicted Class Index: {class_idx} (Confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()
