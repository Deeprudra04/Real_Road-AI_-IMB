#!/usr/bin/env python3
"""
Startup script for Rice Disease Detection Flask App
This script tests the model first and then starts the Flask application
"""

import os
import sys
import subprocess
import time

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'flask',
        'tensorflow', 
        'numpy',
        'Pillow',
        'werkzeug'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - MISSING")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_model_file():
    """Check if the model file exists"""
    print("\n🔍 Checking model file...")
    
    if os.path.exists('rice_disease_model.h5'):
        file_size = os.path.getsize('rice_disease_model.h5') / (1024 * 1024)  # MB
        print(f"✅ Model file found: rice_disease_model.h5 ({file_size:.1f} MB)")
        return True
    else:
        print("❌ Model file not found: rice_disease_model.h5")
        print("Please run: python train_model.py")
        return False

def test_model():
    """Test the model using the test script"""
    print("\n🔍 Testing model functionality...")
    
    try:
        result = subprocess.run([sys.executable, 'test_model.py'], 
                              capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✅ Model test passed")
            return True
        else:
            print("❌ Model test failed")
            print("Error output:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Model test timed out")
        return False
    except Exception as e:
        print(f"❌ Error running model test: {e}")
        return False

def start_flask_app():
    """Start the Flask application"""
    print("\n🚀 Starting Flask application...")
    print("The application will be available at: http://localhost:5000")
    print("Health check: http://localhost:5000/health")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        # Start the Flask app
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\n\n👋 Server stopped by user")
    except Exception as e:
        print(f"\n❌ Error starting Flask app: {e}")

def main():
    """Main startup function"""
    print("🌾 Rice Disease Detection - Startup")
    print("=" * 50)
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\n❌ Cannot start - missing dependencies")
        return
    
    # Step 2: Check model file
    if not check_model_file():
        print("\n❌ Cannot start - model file missing")
        return
    
    # Step 3: Test model
    if not test_model():
        print("\n❌ Cannot start - model test failed")
        return
    
    # Step 4: Start Flask app
    print("\n🎉 All checks passed! Starting Flask application...")
    time.sleep(2)
    start_flask_app()

if __name__ == "__main__":
    main() 