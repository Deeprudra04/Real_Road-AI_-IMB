from flask import Flask, render_template, request, jsonify, url_for
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import base64
import io
import json
import logging
import traceback
from werkzeug.utils import secure_filename

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'JPG', 'JPEG', 'PNG'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load disease classes from saved file
def load_disease_classes():
    try:
        with open('disease_classes.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback disease classes
        logger.info("disease_classes.json not found, using default classes")
        return [
            'bacterial_leaf_blight',
            'brown_spot', 
            'healthy',
            'leaf_blast',
            'leaf_scald',
            'narrow_brown_spot'
        ]

DISEASE_CLASSES = load_disease_classes()

# Load the trained model
def load_model():
    try:
        # Try to load the saved model
        model = tf.keras.models.load_model('rice_disease_model.h5', compile=False)
        logger.info("✅ Model loaded successfully from rice_disease_model.h5")
        
        # Compile the model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        return model
    except FileNotFoundError:
        logger.error("❌ Model file 'rice_disease_model.h5' not found!")
        logger.info("Please run 'python train_model.py' first to train and save the model.")
        return None
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
        logger.error(traceback.format_exc())
        return None

model = load_model()

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for model prediction with better error handling"""
    try:
        logger.info(f"Processing image: {image_path}")
        
        # Load and resize image
        img = load_img(image_path, target_size=(224, 224), color_mode='rgb')
        img_array = img_to_array(img)
        
        # Normalize pixel values
        img_array = img_array / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.info(f"Image processed successfully. Shape: {img_array.shape}")
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        logger.error(traceback.format_exc())
        return None

def predict_disease(image_path):
    """Predict disease from image with comprehensive error handling"""
    if model is None:
        logger.error("Model not loaded")
        return {"error": "Model not loaded. Please ensure the model file exists and is valid."}
    
    try:
        logger.info(f"Starting prediction for: {image_path}")
        
        # Preprocess image
        processed_img = preprocess_image(image_path)
        if processed_img is None:
            return {"error": "Failed to process image. Please ensure the image is valid."}
        
        # Make prediction
        logger.info("Making prediction...")
        predictions = model.predict(processed_img, verbose=0)
        
        # Get predicted class and confidence
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        # Get disease name
        if predicted_class < len(DISEASE_CLASSES):
            disease_name = DISEASE_CLASSES[predicted_class]
        else:
            disease_name = "unknown"
        
        # Get all class probabilities
        class_probabilities = {}
        for i, prob in enumerate(predictions[0]):
            if i < len(DISEASE_CLASSES):
                class_probabilities[DISEASE_CLASSES[i]] = float(prob)
        
        logger.info(f"Prediction successful: {disease_name} with {confidence:.4f} confidence")
        
        return {
            "disease": disease_name,
            "confidence": confidence,
            "all_probabilities": class_probabilities,
            "predicted_class": int(predicted_class)
        }
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        logger.error(traceback.format_exc())
        return {"error": f"Prediction failed: {str(e)}"}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/rice-disease/predict', methods=['POST'])
def predict():
    """Handle image upload and prediction with comprehensive error handling"""
    try:
        logger.info("Received prediction request")
        
        # Check if file was uploaded
        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        
        # Check if file was selected
        if file.filename == '':
            logger.error("No file selected")
            return jsonify({'error': 'No file selected'})
        
        # Check if file type is allowed
        if not allowed_file(file.filename):
            logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type. Please upload an image file (PNG, JPG, JPEG, GIF, BMP)'})
        
        # Secure the filename and save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        logger.info(f"Saving file to: {filepath}")
        file.save(filepath)
        
        # Verify file was saved
        if not os.path.exists(filepath):
            logger.error("File was not saved properly")
            return jsonify({'error': 'Failed to save uploaded file'})
        
        # Make prediction
        logger.info("Starting disease prediction")
        result = predict_disease(filepath)
        
        if 'error' in result:
            logger.error(f"Prediction error: {result['error']}")
            return jsonify(result)
        
        # Add file path for display
        result['image_path'] = url_for('static', filename=f'uploads/{filename}')
        
        logger.info("Prediction completed successfully")
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Server error in predict route: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Server error: {str(e)}'})

@app.route('/predict', methods=['POST'])
def predict_legacy():
    """Legacy route for backward compatibility"""
    return predict()

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'disease_classes': DISEASE_CLASSES
    })

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Maximum size is 16MB'}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({'error': 'Internal server error occurred'}), 500

if __name__ == '__main__':
    if model is None:
        print("\n❌ Cannot start Flask app without trained model!")
        print("Please run: python train_model.py")
        print("Then run: python app.py")
    else:
        print("\n🌐 Starting Flask web application...")
        print("The application will be available at: http://localhost:5000")
        print("Health check: http://localhost:5000/health")
        app.run(debug=True, host='0.0.0.0', port=5000) 