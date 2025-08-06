# Rice Disease Detection - BharatKrishi AI

An AI-powered web application for detecting diseases in rice leaves using deep learning.

## Features

- 🌾 **Rice Disease Detection**: Identifies 6 different rice leaf conditions
- 🤖 **AI-Powered**: Uses DenseNet121 deep learning model
- 🌐 **Web Interface**: Beautiful, responsive web UI
- 📱 **Mobile Friendly**: Works on all devices
- 🔍 **Real-time Analysis**: Instant disease detection results

## Supported Diseases

1. **Bacterial Leaf Blight** - Xanthomonas oryzae pv. oryzae
2. **Brown Spot** - Cochliobolus miyabeanus
3. **Healthy** - No disease detected
4. **Leaf Blast** - Magnaporthe oryzae
5. **Leaf Scald** - Microdochium oryzae
6. **Narrow Brown Spot** - Cercospora janseana

## Quick Start

### Option 1: Easy Startup (Recommended)
```bash
python start_app.py
```

This will:
- ✅ Check all dependencies
- ✅ Verify model file exists
- ✅ Test model functionality
- ✅ Start the Flask application

### Option 2: Manual Startup
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Test the model (optional but recommended)
python test_model.py

# 3. Start the Flask application
python app.py
```

## Access the Application

Once started, open your web browser and go to:
- **Main Application**: http://localhost:5000
- **Health Check**: http://localhost:5000/health

## How to Use

1. **Upload Image**: Click "Choose Image" or drag & drop a rice leaf image
2. **Wait for Analysis**: The AI will process your image
3. **View Results**: See the detected disease and confidence level
4. **Clear**: Click "Clear" to upload a new image

## File Structure

```
Rice Disease Detection/
├── app.py                 # Main Flask application
├── start_app.py          # Startup script with checks
├── test_model.py         # Model testing script
├── train_model.py        # Model training script
├── rice_disease_model.h5 # Trained model file
├── requirements.txt      # Python dependencies
├── rice_data/           # Training dataset
├── static/              # Static files (uploads, CSS, JS)
└── templates/           # HTML templates
```

## Troubleshooting

### Common Issues

1. **"Model not loaded" error**
   - Run: `python train_model.py` to train the model
   - Ensure `rice_disease_model.h5` exists

2. **"An error occurred while processing the image"**
   - Use the new `start_app.py` script
   - Check that all dependencies are installed
   - Verify image format (PNG, JPG, JPEG, GIF, BMP)

3. **Port 5000 already in use**
   - Change port in `app.py` line 156
   - Or kill the process using port 5000

4. **Missing dependencies**
   - Run: `pip install -r requirements.txt`
   - For GPU support: `pip install tensorflow-gpu`

### Health Check

Visit http://localhost:5000/health to verify:
- ✅ Server status
- ✅ Model loading status
- ✅ Available disease classes

## Technical Details

- **Framework**: Flask 2.3.3
- **AI Model**: DenseNet121 (pre-trained on ImageNet)
- **Image Size**: 224x224 pixels
- **Supported Formats**: PNG, JPG, JPEG, GIF, BMP
- **Max File Size**: 16MB

## Development

### Training New Model
```bash
python train_model.py
```

### Testing Model
```bash
python test_model.py
```

### Running Tests
```bash
python start_app.py
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is part of the BharatKrishi AI initiative for Indian agriculture.

---

**Powered by BharatKrishi AI - Team Real_Road AI** 🌾🤖 