# Unified Flask Server for BharatKrishi AI
# Integrates: Crop Prediction, Krishi BOT, and Rice Disease Detection

from flask import Flask, request, jsonify, render_template, send_from_directory, url_for
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import requests
import joblib
import os
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import base64
import io
import json
import sys

# Add Krishi BOT directory to Python path
sys.path.append('Krishi BOT')

# Import Krishi BOT modules with better error handling
KRISHI_BOT_AVAILABLE = False
krishi_chain = None
fallback_chain = None

try:
    print("🔍 Checking Krishi BOT modules...")
    
    # Check if required files exist
    required_files = [
        "Krishi BOT/chat1.py",
        "Krishi BOT/chat2.py", 
        "Krishi BOT/config.py",
        "Krishi BOT/Data/Farming Schemes.pdf",
        "Krishi BOT/Data/farmerbook.pdf"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing Krishi BOT files: {missing_files}")
        raise ImportError(f"Missing required files: {missing_files}")
    
    # Import modules
    from chat1 import fetch_website_content, extract_pdf_text, initialize_vector_store
    from chat2 import llm, setup_retrieval_qa
    from config import GEMINI_API_KEY
    
    print("✅ Krishi BOT modules imported successfully")
    KRISHI_BOT_AVAILABLE = True
    
except ImportError as e:
    print(f"❌ Error importing Krishi BOT modules: {e}")
    KRISHI_BOT_AVAILABLE = False
except Exception as e:
    print(f"❌ Unexpected error with Krishi BOT: {e}")
    KRISHI_BOT_AVAILABLE = False

# Import fallback modules
try:
    from fallback_chat import create_fallback_chain
    from fallback_data import AGRICULTURAL_DATA
    print("✅ Fallback Krishi BOT modules imported successfully")
    fallback_chain = create_fallback_chain()
except Exception as e:
    print(f"⚠️ Warning: Fallback modules not available: {e}")
    fallback_chain = None

# ------------------------------
# 🔁 Define Model Class (Crop Prediction)
# ------------------------------
class CropModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 22)

    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc3(x))
        return self.out(x)

# ------------------------------
# 🚀 Flask App Setup
# ------------------------------
app = Flask(__name__)
CORS(app)

# Configure upload folder for Rice Disease Detection
UPLOAD_FOLDER = 'Rice Disease Detection/static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ------------------------------
# 🧠 Load Models and Data
# ------------------------------

# Crop Prediction Model
print("Loading Crop Prediction model...")
try:
    crop_model = CropModel()
    crop_model.load_state_dict(torch.load("Crop Prediction/model.pt", map_location=torch.device("cpu")))
    crop_model.eval()
    crop_le = joblib.load("Crop Prediction/label_encoder.pkl")
    latlon_df = pd.read_csv("Crop Prediction/location_latlon.csv")
    CROP_PREDICTION_AVAILABLE = True
    print("✅ Crop Prediction model loaded successfully")
except Exception as e:
    print(f"❌ Error loading Crop Prediction model: {e}")
    CROP_PREDICTION_AVAILABLE = False

# Rice Disease Detection Model
print("Loading Rice Disease Detection model...")
try:
    def load_disease_classes():
        try:
            with open('Rice Disease Detection/disease_classes.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print("disease_classes.json not found, using default classes")
            return [
                'bacterial_leaf_blight', 'brown_spot', 'healthy',
                'leaf_blast', 'leaf_scald', 'narrow_brown_spot'
            ]

    DISEASE_CLASSES = load_disease_classes()
    
    # Load the model using the corrected approach from app.py
    try:
        print("🔄 Loading Rice Disease Detection model...")
        rice_model = tf.keras.models.load_model('Rice Disease Detection/rice_disease_model.h5', compile=False)
        print("✅ Model loaded successfully from rice_disease_model.h5")
        
        # Compile the model
        rice_model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Model compiled successfully")
        
    except FileNotFoundError:
        print("❌ Model file 'rice_disease_model.h5' not found!")
        raise Exception("Model file not found")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        raise e
    
    RICE_DISEASE_AVAILABLE = True
    print("✅ Rice Disease Detection model loaded successfully")
except Exception as e:
    print(f"❌ Error loading Rice Disease Detection model: {e}")
    RICE_DISEASE_AVAILABLE = False

# Krishi BOT Initialization
print("Initializing Krishi BOT...")
if KRISHI_BOT_AVAILABLE:
    try:
        print("📚 Loading agricultural data...")
        
        # Test internet connectivity first
        internet_available = False
        try:
            print("🌐 Testing internet connectivity...")
            response = requests.get("https://www.google.com", timeout=5)
            if response.status_code == 200:
                internet_available = True
                print("✅ Internet connection available")
            else:
                print("⚠️ Internet connection test failed")
        except Exception as e:
            print(f"❌ Internet connection not available: {e}")
        
        if internet_available:
            # Example URLs and PDF files
            urls = ["https://mospi.gov.in/4-agricultural-statistics"]
            pdf_files = ["Krishi BOT/Data/Farming Schemes.pdf", "Krishi BOT/Data/farmerbook.pdf"]

            # Fetch content from websites
            website_contents = []
            for url in urls:
                try:
                    print(f"🌐 Fetching content from {url}...")
                    content = fetch_website_content(url)
                    if content:
                        website_contents.append(content)
                        print(f"✅ Successfully fetched content from {url}")
                    else:
                        print(f"⚠️ No content fetched from {url}")
                        website_contents.append("")
                except Exception as e:
                    print(f"❌ Error fetching from {url}: {e}")
                    website_contents.append("")

            # Extract text from PDF files
            pdf_texts = []
            for pdf_file in pdf_files:
                try:
                    if os.path.exists(pdf_file):
                        print(f"📄 Extracting text from {pdf_file}...")
                        text = extract_pdf_text(pdf_file)
                        if text:
                            pdf_texts.append(text)
                            print(f"✅ Successfully extracted text from {pdf_file}")
                        else:
                            print(f"⚠️ No text extracted from {pdf_file}")
                            pdf_texts.append("")
                    else:
                        print(f"❌ PDF file not found: {pdf_file}")
                        pdf_texts.append("")
                except Exception as e:
                    print(f"❌ Error extracting from {pdf_file}: {e}")
                    pdf_texts.append("")

            # Combine all content and initialize
            all_contents = website_contents + pdf_texts
            valid_contents = [content for content in all_contents if content and content.strip()]
            
            if valid_contents:
                print(f"📊 Initializing vector store with {len(valid_contents)} content sources...")
                db = initialize_vector_store(valid_contents)
                
                if db is not None:
                    print("🔗 Setting up retrieval QA chain...")
                    krishi_chain = setup_retrieval_qa(db)
                    
                    if krishi_chain is not None:
                        print("✅ Krishi BOT initialized successfully with online data!")
                    else:
                        print("❌ Failed to setup retrieval QA chain, using fallback...")
                        krishi_chain = fallback_chain
                else:
                    print("❌ Failed to initialize vector store, using fallback...")
                    krishi_chain = fallback_chain
            else:
                print("❌ No valid content found, using fallback...")
                krishi_chain = fallback_chain
        else:
            print("🌐 No internet connection, using fallback Krishi BOT...")
            krishi_chain = fallback_chain
            
    except Exception as e:
        print(f"❌ Error initializing Krishi BOT: {e}")
        print("🔄 Falling back to local agricultural data...")
        krishi_chain = fallback_chain
else:
    print("❌ Krishi BOT modules not available, using fallback...")
    krishi_chain = fallback_chain

# Ensure we have at least the fallback
if krishi_chain is None and fallback_chain is not None:
    print("🔄 Using fallback Krishi BOT...")
    krishi_chain = fallback_chain

# ------------------------------
# 🌦️ Weather API (Crop Prediction)
# ------------------------------
OPENWEATHER_API_KEY = "5a5d56bc65623654441af5a2f29a63d8"

def get_weather(state, district):
    if not CROP_PREDICTION_AVAILABLE:
        return None, None
    
    row = latlon_df[
        (latlon_df['State'].str.lower() == state.lower()) &
        (latlon_df['District'].str.lower() == district.lower())
    ]
    if row.empty:
        return None, None

    lat, lon = float(row['Latitude'].iloc[0]), float(row['Longitude'].iloc[0])
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
    response = requests.get(url)
    if response.status_code != 200:
        return None, None
    data = response.json()
    temp = data["main"]["temp"]
    humidity = data["main"]["humidity"]
    return temp, humidity

# ------------------------------
# 🛣️ Routes
# ------------------------------

@app.route("/")
def home():
    """Main landing page"""
    return send_from_directory(".", "index.html")

@app.route("/static/<path:filename>")
def static_files(filename):
    """Serve static files"""
    return send_from_directory("static", filename)

# Crop Prediction Routes
@app.route("/crop-prediction")
def crop_prediction_page():
    """Crop Prediction service page"""
    return send_from_directory("Crop Prediction", "index.html")

@app.route("/crop-prediction/dashboard")
def crop_dashboard():
    """Crop Prediction dashboard"""
    return send_from_directory("Crop Prediction", "dashboard.html")

@app.route("/crop-prediction/predict", methods=["POST"])
def crop_predict():
    """Crop prediction API endpoint"""
    if not CROP_PREDICTION_AVAILABLE:
        return jsonify({"error": "Crop prediction service is not available"}), 503
    
    data = request.json
    try:
        N = float(data.get("N"))
        P = float(data.get("P"))
        K = float(data.get("K"))
        pH = float(data.get("ph"))
        state = data.get("state")
        district = data.get("district")
        month = data.get("month")

        # Get temperature and humidity via API
        temp, humidity = get_weather(state, district)
        if temp is None or humidity is None:
            return jsonify({"error": "Weather data not available for this location"}), 400

        rainfall = 80.0
        input_vals = [N, P, K, temp, humidity, pH, rainfall]
        x = torch.tensor([input_vals], dtype=torch.float32)

        # Model prediction
        with torch.no_grad():
            out = crop_model(x)
            pred_idx = torch.argmax(out, dim=1).item()
            crop_name = crop_le.inverse_transform([pred_idx])[0]

        return jsonify({"prediction": crop_name})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Krishi BOT Routes
@app.route("/krishi-bot")
def krishi_bot_page():
    """Krishi BOT service page"""
    return send_from_directory("Krishi BOT/templates", "index.html")

@app.route("/krishi-bot/static/<path:filename>")
def krishi_bot_static(filename):
    """Serve Krishi BOT static files"""
    return send_from_directory("Krishi BOT/static", filename)

@app.route("/krishi-bot/ask", methods=["POST"])
def krishi_bot_ask():
    """Krishi BOT chat API endpoint"""
    if krishi_chain is None:
        return jsonify({"answer": "Sorry, the Krishi BOT is not properly initialized. Please check the console for errors or wait a moment and try again."})
    
    try:
        query = request.form['messageText'].strip()
        
        if not query:
            return jsonify({"answer": "Please provide a question to ask."})

        if query.lower() in ["who developed you?", "who created you?", "who made you?"]:
            return jsonify({"answer": "I was developed by the Real_Road AI team for BharatKrishi AI."})
        
        print(f"🤖 Processing query: {query}")
        
        # Check if this is a fallback response
        is_fallback = krishi_chain == fallback_chain
        
        response = krishi_chain(query)
        
        if response and 'result' in response:
            answer = response['result']
            if is_fallback:
                answer += "\n\n💡 Note: This response is from local agricultural data since internet access is limited."
            print(f"✅ Response generated: {answer[:100]}...")
            return jsonify({"answer": answer})
        else:
            print(f"❌ No response generated for query: {query}")
            return jsonify({"answer": "I don't have information about that specific topic. Please try asking about agriculture, farming practices, or crop management."})
            
    except Exception as e:
        print(f"❌ Error processing query: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"answer": "Sorry, there was an error processing your request. Please try again."})

# Rice Disease Detection Routes
@app.route("/rice-disease")
def rice_disease_page():
    """Rice Disease Detection service page"""
    return send_from_directory("Rice Disease Detection/templates", "index.html")

@app.route("/rice-disease/static/<path:filename>")
def rice_disease_static(filename):
    """Serve Rice Disease Detection static files"""
    return send_from_directory("Rice Disease Detection/static", filename)

@app.route("/rice-disease/about")
def rice_disease_about():
    """Rice Disease Detection about page"""
    return send_from_directory("Rice Disease Detection/templates", "about.html")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    """Preprocess image for model prediction"""
    try:
        img = load_img(image_path, target_size=(224, 224), color_mode='rgb')
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

def predict_disease(image_path):
    """Predict disease from image"""
    if not RICE_DISEASE_AVAILABLE:
        return {"error": "Rice disease detection model not loaded."}
    
    try:
        processed_img = preprocess_image(image_path)
        if processed_img is None:
            return {"error": "Failed to process image"}
        
        predictions = rice_model.predict(processed_img)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        disease_name = DISEASE_CLASSES[predicted_class]
        
        class_probabilities = {}
        for i, prob in enumerate(predictions[0]):
            class_probabilities[DISEASE_CLASSES[i]] = float(prob)
        
        return {
            "disease": disease_name,
            "confidence": confidence,
            "all_probabilities": class_probabilities,
            "predicted_class": int(predicted_class)
        }
        
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

@app.route("/rice-disease/predict", methods=["POST"])
def rice_disease_predict():
    """Rice disease prediction API endpoint"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'})
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'})
        
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            result = predict_disease(filepath)
            
            if 'error' in result:
                return jsonify(result)
            
            result['image_path'] = url_for('rice_disease_static', filename=f'uploads/{filename}')
            
            return jsonify(result)
        else:
            return jsonify({'error': 'Invalid file type'})
            
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'})

# Service Status API
@app.route("/api/status")
def service_status():
    """API endpoint to check service availability"""
    return jsonify({
        "crop_prediction": CROP_PREDICTION_AVAILABLE,
        "krishi_bot": krishi_chain is not None,
        "rice_disease_detection": RICE_DISEASE_AVAILABLE,
        "server": "running"
    })

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🌱 BharatKrishi AI - Unified Server")
    print("="*60)
    print(f"✅ Crop Prediction: {'Available' if CROP_PREDICTION_AVAILABLE else 'Not Available'}")
    print(f"✅ Krishi BOT: {'Available' if krishi_chain is not None else 'Not Available'}")
    print(f"✅ Rice Disease Detection: {'Available' if RICE_DISEASE_AVAILABLE else 'Not Available'}")
    print("="*60)
    print("🌐 Server will be available at: http://localhost:5000")
    print("📱 Services:")
    print("   • Main Page: http://localhost:5000/")
    print("   • Crop Prediction: http://localhost:5000/crop-prediction")
    print("   • Krishi BOT: http://localhost:5000/krishi-bot")
    print("   • Rice Disease Detection: http://localhost:5000/rice-disease")
    print("="*60)
    
    app.run(debug=True, host='0.0.0.0', port=5000) 