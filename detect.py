import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
# Load the trained model
model = load_model("./model/conv2d-lung_cancer_detection-99.68.h5")

# ✅ Class labels must match training order
CLASS_LABELS = [
    "Colon Adenocarcinoma", 
    "Colon Benign Tissue", 
    "Lung Adenocarcinoma", 
    "Lung Benign Tissue", 
    "Lung Squamous Cell Carcinoma"
]

def predict(img_path):
    """Predict class for a given image path."""
    # Load image exactly as in training
    img = image.load_img(img_path, target_size=(224, 224), color_mode='rgb')
    img_array = image.img_to_array(img)  # do NOT divide by 255
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    predicted_class = np.argmax(preds[0])
    confidence = np.max(preds[0])

    print("Raw predictions:", preds[0])
    print("Sum of probs:", np.sum(preds[0]))

    return CLASS_LABELS[predicted_class], confidence

if __name__ == "__main__":
    test_image = "./lungscc100.jpeg"  # replace with your image path
    label, conf = predict(test_image)
    print(f"Predicted: {label} (confidence: {conf:.2f})")
