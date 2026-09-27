import sys
import cv2
import numpy as np
import tensorflow as tf

def predict_image(image_path, model_path='cat_dog_model.keras', class_names=['cat', 'dog']):
    """Loads trained model and predicts class for a single image file."""
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model from {model_path}. Train the model first via train.py!")
        return

    # Preprocessing pipeline
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from path '{image_path}'")
        return
        
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (256, 256))
    input_tensor = img_resized.reshape((1, 256, 256, 3)) / 255.0

    # Inference
    prob = model.predict(input_tensor, verbose=0)[0][0]
    
    if prob >= 0.5:
        print(f"\nResult: {class_names[1].upper()} (Confidence: {prob*100:.2f}%)")
    else:
        print(f"\nResult: {class_names[0].upper()} (Confidence: {(1-prob)*100:.2f}%)")

if __name__ == '__main__':
    # Accept image path from command line arguments or prompt user
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    else:
        img_path = input("Enter path to test image (e.g., dog.jpg): ").strip()
        
    predict_image(img_path)