import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load saved model and label mapping
def load_artifacts(model_path="models/skin_disease_model.h5", label_map_path="models/label_map.npy"):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    model = load_model(model_path)
    
    if os.path.exists(label_map_path):
        label_map = np.load(label_map_path, allow_pickle=True).item()
    else:
        label_map = None
    return model, label_map


# Preprocess uploaded image for prediction
def preprocess_pil(pil_img, target_size=(224, 224)):
    img = pil_img.convert("RGB").resize(target_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Run prediction
def predict_pil(model, pil_img, label_map=None):
    img_array = preprocess_pil(pil_img)
    preds = model.predict(img_array)
    pred_idx = np.argmax(preds, axis=1)[0]
    confidence = float(np.max(preds))
    
    if label_map:
        label = label_map.get(pred_idx, str(pred_idx))
    else:
        label = str(pred_idx)
    return label, confidence
