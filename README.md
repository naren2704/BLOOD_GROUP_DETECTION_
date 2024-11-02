# Blood Group Detection from Fingerprint Images

This project uses a Convolutional Neural Network (CNN) to detect blood groups based on fingerprint images. The model is integrated into a user-friendly web application where users can upload their fingerprint images to receive predictions.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Directory Structure](#directory-structure)
- [How to Run the Project](#how-to-run-the-project)
- [Future Enhancements](#future-enhancements)
- [Contributors](#contributors)

## Overview
Blood group detection using fingerprint patterns is an emerging technology that provides a quick and non-invasive way to identify blood types. This project leverages deep learning, specifically CNNs, to analyze fingerprint images and predict the corresponding blood group. 

The main aim is to facilitate easier and faster blood group identification through image processing techniques.

## Features
- User-friendly web interface to upload fingerprint images
- CNN model for accurate blood group classification
- Real-time results displayed on the web application
- Error handling for incorrect file formats and unsupported images

## Technologies Used
- **Frontend**: HTML, CSS
- **Backend**: Flask
- **Machine Learning**: Convolutional Neural Network (MobileNetV2) for blood group detection
- **Database**: SQLite (for user credentials and uploads)
- **Deployment**: (Specify if using Heroku, AWS, etc., or leave blank if running locally)

## Directory Structure
Here's an overview of the project structure:
blood_group_detection_app/ ├── app.py # Flask app code ├── model/ │ └── fingerprint_blood_group_model.h5 # Trained CNN model ├── static/ │ ├── styles.css # CSS styles │ └── uploads/ # Uploaded images ├── templates/ │ ├── home.html # Home page │ ├── upload.html # Upload page │ └── result.html # Result page └── README.md # Project documentation

## How to Run the Project
To run this project locally, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/blood-group-detection.git
   cd blood-group-detection
Install required dependencies: Make sure you have Python installed. Then run:
pip install -r requirements.txt

Run the Flask app:
python app.py
Access the Web Interface: Open your browser and go to http://127.0.0.1:5000 to use the application.

Future Enhancements
Improve model accuracy with a larger dataset
Add support for different types of biometric input
Integrate user authentication and authorization
Deploy the application for public use
Contributors
Alagu Nachiyar K
Vaishali
Narendran

