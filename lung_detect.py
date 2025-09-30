import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os
import sys

# Load the trained lung cancer detection model from model2 folder
MODEL_PATH = "./model2/conv2d-lung_detection-98.73.h5"

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    sys.exit(1)

print("Loading lung cancer detection model...")
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Class labels for lung cancer detection (based on lung_detection-class_dict.csv)
CLASS_LABELS = [
    "Lung Adenocarcinoma",      # Malignant - Type of lung cancer
    "Lung Benign Tissue",       # Benign - Normal/healthy lung tissue  
    "Lung Squamous Cell Carcinoma"  # Malignant - Another type of lung cancer
]

# Cancer classification mapping
CANCER_STATUS = {
    "Lung Adenocarcinoma": "CANCER DETECTED",
    "Lung Benign Tissue": "NO CANCER",
    "Lung Squamous Cell Carcinoma": "CANCER DETECTED"
}

def preprocess_image(img_path):
    """
    Preprocess image for lung cancer prediction.
    
    Args:
        img_path (str): Path to the image file
        
    Returns:
        numpy.ndarray: Preprocessed image array ready for prediction
    """
    try:
        # Load image with target size matching training data (224x224)
        img = image.load_img(img_path, target_size=(224, 224), color_mode='rgb')
        
        # Convert to array
        img_array = image.img_to_array(img)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Note: Based on the existing detect.py, no normalization is applied
        # Keep pixel values in range [0, 255] as per training
        
        return img_array
        
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_lung_cancer(img_path, show_details=True):
    """
    Predict if the lung image shows cancer or not.
    
    Args:
        img_path (str): Path to the lung image
        show_details (bool): Whether to show detailed prediction results
        
    Returns:
        tuple: (predicted_class, confidence, cancer_status)
    """
    # Check if image file exists
    if not os.path.exists(img_path):
        print(f"Error: Image file not found at {img_path}")
        return None, None, None
    
    # Preprocess the image
    img_array = preprocess_image(img_path)
    if img_array is None:
        return None, None, None
    
    try:
        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        
        # Get the predicted class and confidence
        predicted_class_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        predicted_class = CLASS_LABELS[predicted_class_idx]
        cancer_status = CANCER_STATUS[predicted_class]
        
        if show_details:
            print(f"\n{'='*50}")
            print(f"LUNG CANCER DETECTION RESULTS")
            print(f"{'='*50}")
            print(f"Image: {os.path.basename(img_path)}")
            print(f"Predicted Class: {predicted_class}")
            print(f"Confidence: {confidence:.4f} ({confidence*100:.2f}%)")
            print(f"Cancer Status: {cancer_status}")
            print(f"{'='*50}")
            
            # Show all class probabilities
            print(f"\nDetailed Predictions:")
            for i, (class_name, prob) in enumerate(zip(CLASS_LABELS, predictions[0])):
                status = CANCER_STATUS[class_name]
                print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%) - {status}")
            print(f"{'='*50}\n")
        
        return predicted_class, confidence, cancer_status
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None, None

def batch_predict(image_folder):
    """
    Predict lung cancer for all images in a folder.
    
    Args:
        image_folder (str): Path to folder containing lung images
    """
    if not os.path.exists(image_folder):
        print(f"Error: Folder not found at {image_folder}")
        return
    
    # Supported image extensions
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
    
    # Get all image files
    image_files = [f for f in os.listdir(image_folder) 
                   if f.lower().endswith(supported_extensions)]
    
    if not image_files:
        print(f"No supported image files found in {image_folder}")
        return
    
    print(f"Found {len(image_files)} images. Starting batch prediction...\n")
    
    results = []
    for img_file in image_files:
        img_path = os.path.join(image_folder, img_file)
        predicted_class, confidence, cancer_status = predict_lung_cancer(img_path, show_details=False)
        
        if predicted_class is not None:
            results.append({
                'image': img_file,
                'prediction': predicted_class,
                'confidence': confidence,
                'cancer_status': cancer_status
            })
            print(f"✓ {img_file}: {cancer_status} ({confidence*100:.1f}%)")
        else:
            print(f"✗ {img_file}: Failed to process")
    
    # Summary
    print(f"\n{'='*60}")
    print(f"BATCH PREDICTION SUMMARY")
    print(f"{'='*60}")
    cancer_count = sum(1 for r in results if "CANCER DETECTED" in r['cancer_status'])
    benign_count = len(results) - cancer_count
    print(f"Total Images Processed: {len(results)}")
    print(f"Cancer Detected: {cancer_count}")
    print(f"No Cancer (Benign): {benign_count}")
    print(f"{'='*60}")

if __name__ == "__main__":
    print("Lung Cancer Detection System")
    print("Model: conv2d-lung_detection-98.73.h5")
    print("Classes: Lung Adenocarcinoma, Lung Benign Tissue, Lung Squamous Cell Carcinoma\n")
    
    # Example usage - replace with your image path
    test_image = "./sample images/lungaca10.jpeg"  # Replace with actual image path
    
    # Check if test image exists, if not provide instructions
    if os.path.exists(test_image):
        print(f"Testing with image: {test_image}")
        predicted_class, confidence, cancer_status = predict_lung_cancer(test_image)
    else:
        print(f"Test image not found at: {test_image}")
        print("\nTo use this script:")
        print("1. Single image prediction:")
        print("   python lung_detect.py")
        print("   (Update the test_image variable with your image path)")
        print("\n2. In your Python code:")
        print("   from lung_detect import predict_lung_cancer")
        print("   result = predict_lung_cancer('path/to/your/image.jpg')")
        print("\n3. Batch prediction:")
        print("   from lung_detect import batch_predict")
        print("   batch_predict('path/to/image/folder')")
        
        print(f"\nSupported image formats: .jpg, .jpeg, .png, .bmp, .tiff, .tif")
        print(f"Image preprocessing: Resized to 224x224 pixels, RGB color mode")
