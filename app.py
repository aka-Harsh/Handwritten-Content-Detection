# app.py - Flask API for Handwriting Recognition with EasyOCR

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import numpy as np
import cv2
import base64
import io
import os
import time
from PIL import Image

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variable for the OCR reader
reader = None

def initialize_ocr():
    """Initialize the EasyOCR reader"""
    global reader
    try:
        print("Initializing EasyOCR (this may take a minute the first time)...")
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='./ocr_models')
        print("EasyOCR initialized successfully!")
    except Exception as e:
        print(f"Error initializing EasyOCR: {str(e)}")
        print("Attempting to install EasyOCR...")
        os.system('pip install easyocr')
        
        # Retry initialization
        import easyocr
        reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='./ocr_models')
        print("EasyOCR installed and initialized successfully!")

def preprocess_image(image_data):
    """Process image data for better OCR recognition"""
    # Convert base64 to image
    image = Image.open(io.BytesIO(image_data))
    image = np.array(image)
    
    # Convert to grayscale if it's a color image
    if len(image.shape) == 3 and image.shape[2] > 1:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply thresholding to make it more binary (clearer digit)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Add a white border around the image (helps with OCR)
    image = cv2.copyMakeBorder(image, 20, 20, 20, 20, cv2.BORDER_CONSTANT, value=255)
    
    # Resize to a good size for OCR (not too small)
    image = cv2.resize(image, (200, 200), interpolation=cv2.INTER_AREA)
    
    # Apply slight Gaussian blur to reduce noise
    image = cv2.GaussianBlur(image, (5, 5), 0)
    
    # Increase contrast
    image = cv2.convertScaleAbs(image, alpha=1.5, beta=0)
    
    return image

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions"""
    if request.method == 'POST':
        # Check if OCR reader is initialized
        if reader is None:
            return jsonify({'error': 'OCR reader not initialized'}), 500
        
        # Get the image from the request
        if 'image' not in request.files:
            # Try to get base64 image from JSON
            data = request.get_json()
            if data and 'image' in data:
                # Remove header from base64 string if present
                base64_str = data['image']
                if ',' in base64_str:
                    base64_str = base64_str.split(',')[1]
                
                # Decode base64 string
                image_data = base64.b64decode(base64_str)
            else:
                return jsonify({'error': 'No image provided'}), 400
        else:
            # Get image file
            image_file = request.files['image']
            image_data = image_file.read()
        
        # Preprocess the image
        try:
            processed_image = preprocess_image(image_data)
            
            # Save processed image for debugging (optional)
            cv2.imwrite('processed_image.png', processed_image)
        except Exception as e:
            return jsonify({'error': f'Error processing image: {str(e)}'}), 400
        
        # Recognize the digit using EasyOCR
        start_time = time.time()
        results = reader.readtext(processed_image, allowlist='0123456789', detail=1)
        processing_time = time.time() - start_time
        
        # Process results
        if results:
            # Sort results by confidence (highest first)
            results.sort(key=lambda x: x[2], reverse=True)
            
            # Get the top result
            bbox, text, confidence = results[0]
            
            # Only return single digits
            if len(text) == 1 and text.isdigit():
                digit = int(text)
                return jsonify({
                    'digit': digit,
                    'confidence': float(confidence),
                    'processing_time': processing_time
                })
        
        # If no valid result was found
        return jsonify({
            'digit': 'Unknown',
            'confidence': 0.0,
            'processing_time': processing_time,
            'message': 'Could not detect a single digit in the image'
        })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'ocr_initialized': reader is not None})

if __name__ == '__main__':
    # Initialize OCR
    initialize_ocr()
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
