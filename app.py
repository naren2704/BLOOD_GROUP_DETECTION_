from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'  # Folder for storing uploaded images

# Load the trained MobileNetV2 model
model = load_model(r'C:\Users\SEC\Desktop\project\model\fingerprint_blood_group_model.h5')

# Allowed file types for upload
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg', 'bmp'}

# Function to preprocess fingerprint image before prediction
def preprocess_image(image_path):
    # Load image using OpenCV (RGB)
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # Load as RGB
    
    # Resize image to match the input shape of the model (128x128)
    image = cv2.resize(image, (128, 128))
    
    # Normalize the image (values between 0 and 1)
    image = image.astype('float32') / 255.0
    
    # Expand dimensions to match the model input (1, 128, 128, 3)
    image = np.expand_dims(image, axis=0)
    
    return image

# Home page
@app.route('/')
def home():
    return render_template('home.html')

# Upload page
@app.route('/upload')
def upload():
    return render_template('upload.html')

# Handle image upload and prediction
@app.route('/upload_image', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('upload'))

    file = request.files['file']
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Preprocess the image and predict the blood group
            blood_group, confidence = predict_blood_group(filepath)
            return render_template('result.html', blood_group=blood_group, confidence=confidence, image=filename)
        except ValueError as e:
            print(f"Error: {e}")
            return render_template('upload.html', error=str(e))
    else:
        print("Invalid file type.")
        return redirect(url_for('upload'))

# Function to predict the blood group based on the fingerprint image
def predict_blood_group(image_path):
    # Preprocess the image
    image = preprocess_image(image_path)

    # Make the prediction using the loaded model
    prediction = model.predict(image)
    blood_group_idx = np.argmax(prediction)  # Get the class with the highest probability
    confidence = np.max(prediction)  # Get the confidence of the prediction

    # Mapping class indices to blood group names
    blood_groups = [ 'A+', 'B+', 'AB+', 'O+','A-', 'B-', 'AB-','-O']
    
    if blood_group_idx < len(blood_groups):
        return blood_groups[blood_group_idx], confidence
    else:
        raise ValueError("Predicted index is out of bounds for blood groups.")

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)
