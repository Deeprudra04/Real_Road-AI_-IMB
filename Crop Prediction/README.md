# 🌾 BharatKrishi AI - Smart Crop Recommendation System

## 📋 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Technology Stack](#technology-stack)
- [System Architecture](#system-architecture)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [API Documentation](#api-documentation)
- [Model Details](#model-details)
- [Data Sources](#data-sources)
- [Project Structure](#project-structure)
- [Screenshots](#screenshots)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

**BharatKrishi AI** is an intelligent crop recommendation system that leverages machine learning to provide personalized crop suggestions based on soil parameters, weather conditions, and geographical location. The system is designed to assist Indian farmers in making informed decisions about crop selection, ultimately improving agricultural productivity and sustainability.

### Key Objectives
- Provide accurate crop recommendations based on soil and weather data
- Support 28 Indian states and 150+ districts
- Real-time weather data integration
- User-friendly web interface with green agricultural theme
- Scalable and maintainable architecture

## ✨ Features

### 🌱 Core Features
- **AI-Powered Recommendations**: Neural network model trained on comprehensive crop data
- **Real-time Weather Integration**: Automatic weather data fetching from OpenWeather API
- **Geographic Coverage**: Support for 28 Indian states and 150+ districts
- **Interactive Web Interface**: Modern, responsive design with green agricultural theme
- **API Testing Dashboard**: Built-in testing tools for API validation

### 🎨 User Interface Features
- **Modern Design**: Green color scheme with gradient backgrounds and less contrast
- **Responsive Layout**: Works seamlessly on desktop, tablet, and mobile devices
- **Navigation Bar**: Easy access to different sections (Home, Dashboard)
- **Real-time Feedback**: Loading states and success/error notifications
- **Form Validation**: Input validation with visual feedback

### 📊 Dashboard Features
- **System Status Monitoring**: Real-time status of all system components
- **Statistics Display**: Key metrics including supported crops, states, and districts
- **API Health Checks**: Monitoring of Flask server, ML model, and weather API
- **Location Management**: Comprehensive list of supported locations
- **Simplified Navigation**: Clean interface with Home and Dashboard sections

## 🛠 Technology Stack

### Backend Technologies
- **Python 3.8+**: Core programming language
- **Flask**: Web framework for API development
- **PyTorch**: Deep learning framework for neural network
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Requests**: HTTP library for API calls
- **Joblib**: Model serialization

### Frontend Technologies
- **HTML5**: Semantic markup
- **CSS3**: Styling with modern features (Grid, Flexbox, Gradients)
- **JavaScript (ES6+)**: Client-side interactivity
- **Font Awesome**: Icon library
- **Google Fonts**: Typography (Poppins)

### External APIs
- **OpenWeather API**: Real-time weather data
- **Geographic Data**: Indian states and districts coordinates

### Development Tools
- **Git**: Version control
- **Requirements.txt**: Dependency management
- **CORS**: Cross-origin resource sharing

## 🏗 System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   External      │
│   (HTML/CSS/JS) │◄──►│   (Flask/Python)│◄──►│   APIs          │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Interface│    │   ML Model      │    │   Weather Data  │
│   - Forms       │    │   - Neural Net  │    │   - Temperature │
│   - Dashboard   │    │   - Predictions │    │   - Humidity    │
│   - API Test    │    │   - Data Proc.  │    │   - Location    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Data Flow
1. **User Input**: Soil parameters (N, P, K, pH) and location
2. **Weather Fetch**: Real-time weather data from OpenWeather API
3. **Model Processing**: Neural network processes input features
4. **Prediction**: Crop recommendation based on trained model
5. **Response**: JSON response with recommended crop

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)
- Git (for cloning repository)

### Step 1: Clone Repository
```bash
git clone <repository-url>
cd Crop_recommendation
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: API Key Setup
1. Get a free API key from [OpenWeather](https://openweathermap.org/api)
2. Update the API key in `app.py`:
```python
OPENWEATHER_API_KEY = "your_api_key_here"
```

### Step 4: Run Application
```bash
python app.py
```

### Step 5: Access Application
Open your web browser and navigate to:
- **Home Application**: http://127.0.0.1:5000
- **Dashboard**: http://127.0.0.1:5000/dashboard

## 📖 Usage Guide

### Getting Started
1. **Navigate to Home Page**: Access the crop recommendation form
2. **Enter Soil Parameters**:
   - Nitrogen (N): 0-140
   - Phosphorous (P): 5-145
   - Potassium (K): 5-205
   - pH Level: 3.5-10.0
3. **Select Location**: Choose state and district from dropdown
4. **Choose Month**: Select the planting month
5. **Get Recommendation**: Click "Get Recommendation" button

### Understanding Results
- **Recommended Crop**: AI-suggested crop based on inputs
- **Weather Data**: Current temperature and humidity
- **Confidence**: Model confidence in the recommendation



## 🔌 API Documentation

### Endpoint: `/predict`
**Method**: POST  
**Content-Type**: application/json

#### Request Parameters
```json
{
  "N": 50,           // Nitrogen (0-140)
  "P": 50,           // Phosphorous (5-145)
  "K": 50,           // Potassium (5-205)
  "ph": 6.5,         // pH Level (3.5-10.0)
  "state": "Maharashtra",
  "district": "Mumbai",
  "month": "January"
}
```

#### Response Format
```json
{
  "prediction": "papaya"
}
```

#### Error Response
```json
{
  "error": "Weather data not available for this location"
}
```

### Supported Locations
- **28 Indian States**: Complete coverage of Indian states
- **150+ Districts**: Major agricultural districts
- **Real-time Weather**: Temperature and humidity data

## 🤖 Model Details

### Neural Network Architecture
```
Input Layer (7 features) → Hidden Layer 1 (64 neurons) → 
Hidden Layer 2 (128 neurons) → Hidden Layer 3 (64 neurons) → 
Output Layer (22 crop classes)
```

### Model Features
- **Input Features**: 7 parameters (N, P, K, temperature, humidity, pH, rainfall)
- **Output Classes**: 22 different crop types
- **Activation Function**: SELU (Scaled Exponential Linear Unit)
- **Training Data**: Comprehensive crop dataset with soil and weather parameters

### Model Performance
- **Accuracy**: High accuracy on test dataset
- **Generalization**: Good performance across different regions
- **Scalability**: Efficient inference for real-time predictions

## 📊 Data Sources

### Primary Data Sources
1. **Crop Recommendation Dataset**: Historical crop and soil data
2. **Weather Data**: OpenWeather API for real-time weather
3. **Geographic Data**: Indian states and districts coordinates
4. **Soil Parameters**: NPK values and pH levels

### Data Preprocessing
- **Feature Scaling**: Normalized input parameters
- **Label Encoding**: Categorical crop names to numerical labels
- **Data Validation**: Input range validation
- **Missing Data Handling**: Robust error handling

## 📁 Project Structure

```
Crop_ recomendation/
├── app.py                          # Main Flask application
├── model.pt                        # Trained PyTorch model
├── label_encoder.pkl               # Label encoder for crops
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── index.html                      # Main application interface
├── dashboard.html                  # System dashboard
├── test.html                       # API testing interface
├── data/                           # Data files directory
│   ├── Crop_recommendation.csv     # Training dataset
│   ├── location_latlon.csv         # Geographic coordinates
│   ├── state_latlon.csv           # State coordinates
│   ├── ApportionedIdentifiers.csv # Additional data
│   ├── cities_list.xlsx           # Cities data
│   ├── city_lat.csv               # City coordinates
│   ├── district wise rainfall normal.csv # Rainfall data
│   ├── list_of_latitudelongitude_of_cities_of_india-2049j.csv # Indian cities
│   ├── price.txt                  # Price data
│   ├── rainfall in india 1901-2015.csv # Historical rainfall
│   ├── rainfall_lat_long_fuzzy.csv # Rainfall coordinates
│   ├── rainfall_lat_long.csv      # Rainfall location data
│   ├── response.json              # API responses
│   └── crop/                      # Crop-specific data
└── tempCodeRunnerFile.py          # Temporary development file
```

### Key Files Description
- **app.py**: Main application with Flask routes and ML model
- **model.pt**: Serialized PyTorch neural network model
- **label_encoder.pkl**: Scikit-learn label encoder for crop names
- **index.html**: Main user interface for crop recommendations
- **dashboard.html**: System monitoring and statistics dashboard
- **test.html**: API testing and validation interface

## 🖼 Screenshots

### Home Interface
- Modern green-themed design with gradient backgrounds
- Intuitive form layout with soil parameter inputs
- Real-time weather integration
- Responsive design for all devices
- Navigation bar with BharatKrishi AI branding

### Dashboard
- System status monitoring with green theme
- Statistics and metrics display
- API health checks
- Location management interface
- Simplified navigation with Home and Dashboard sections

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

### Code Standards
- Follow PEP 8 Python style guide
- Add comments for complex logic
- Update documentation for new features
- Maintain consistent code formatting

### Testing
- Test API endpoints
- Validate model predictions
- Check UI responsiveness
- Verify error handling

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **OpenWeather API**: For real-time weather data
- **PyTorch**: For deep learning framework
- **Flask**: For web framework
- **Font Awesome**: For icons
- **Google Fonts**: For typography

## 📞 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation for common issues

## 🔄 Version History

### Version 1.0.0 (Current)
- Initial release with core functionality
- Neural network model implementation
- Web interface with green agricultural theme
- API testing capabilities
- Dashboard for system monitoring
- Support for 28 Indian states and 150+ districts

---

**BharatKrishi AI** - Empowering Indian farmers with intelligent crop recommendations through the power of artificial intelligence and modern web technologies. 