from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.keras.preprocessing import image

# Path to the saved model
model_path = r'C:\Users\SEC\Desktop\archive\fingerprint_blood_group_model.h5'

# Load the saved model
model = load_model(model_path)
print("Model loaded successfully.")

# Function to predict blood group from an image
def predict_blood_group(img_path):
    img = image.load_img(img_path, target_size=(128, 128))  # Resize the image
    img_array = image.img_to_array(img) / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)  # Get the index of the class with highest probability
    
    return predicted_class, predictions

# Example usage:
test_image_path = r'C:\Users\SEC\Desktop\archive\dataset_blood_group\organized\A-\A-_cluster_1_5.jpg'  # Change this to your test image path
predicted_class, predictions = predict_blood_group(test_image_path)
print(f"Predicted class: {predicted_class}")
print(f"Predictions: {predictions}")
