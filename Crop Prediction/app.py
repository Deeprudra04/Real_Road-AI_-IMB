# app.py
from flask import Flask, request, jsonify, render_template_string, send_from_directory
from flask_cors import CORS
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import requests
import joblib
import os

# ------------------------------
# 🔁 Define Model Class (same as Colab)
# ------------------------------
class CropModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(7, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 22)  # update if your model has a different number of crops

    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc3(x))
        return self.out(x)

# ------------------------------
# 🧠 Load Trained Model and Label Encoder
# ------------------------------
model = CropModel()
model.load_state_dict(torch.load("model.pt", map_location=torch.device("cpu")))
model.eval()

le = joblib.load("label_encoder.pkl")

# ------------------------------
# 🌍 Load District-wise Coordinates
# ------------------------------
latlon_df = pd.read_csv("location_latlon.csv")

# ------------------------------
# 🌦️ Get Weather from OpenWeather API
# ------------------------------
OPENWEATHER_API_KEY = "5a5d56bc65623654441af5a2f29a63d8"  # replace with your API key if needed

def get_weather(state, district):
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
# 🚀 Flask App Setup
# ------------------------------
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route("/")
def home():
    return send_from_directory(".", "index.html")

@app.route("/test")
def test():
    return send_from_directory(".", "test.html")

@app.route("/dashboard")
def dashboard():
    return send_from_directory(".", "dashboard.html")



@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    try:
        N = float(data.get("N"))
        P = float(data.get("P"))
        K = float(data.get("K"))
        pH = float(data.get("ph"))  # Changed from "pH" to "ph" to match frontend
        state = data.get("state")
        district = data.get("district")
        month = data.get("month")  # currently unused, but can be used later

        # Get temperature and humidity via API
        temp, humidity = get_weather(state, district)
        if temp is None or humidity is None:
            return jsonify({"error": "Weather data not available for this location"}), 400

        # Optional: fixed rainfall or enhance this later
        rainfall = 80.0

        # Prepare model input
        input_vals = [N, P, K, temp, humidity, pH, rainfall]
        x = torch.tensor([input_vals], dtype=torch.float32)

        # Model prediction
        with torch.no_grad():
            out = model(x)
            pred_idx = torch.argmax(out, dim=1).item()
            crop_name = le.inverse_transform([pred_idx])[0]

        return jsonify({"prediction": crop_name})  # Changed from "recommended_crop" to "prediction"

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
