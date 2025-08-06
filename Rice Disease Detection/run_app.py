#!/usr/bin/env python3
"""
Rice Leaf Disease Detection - Quick Start Script
This script helps you get started with the application.
"""

import os
import sys
import subprocess

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'flask', 'tensorflow', 'numpy', 'PIL', 'sklearn', 'matplotlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("❌ Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print("\nPlease install dependencies:")
        print("pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies are installed")
    return True

def check_dataset():
    """Check if dataset is available"""
    dataset_paths = [
        'rice_data/RiceLeafsDisease',
        'extracted_dataset/RiceLeafsDisease',
        'rice_data.zip'
    ]
    
    for path in dataset_paths:
        if os.path.exists(path):
            print(f"✅ Dataset found at: {path}")
            return True
    
    print("❌ Dataset not found!")
    print("Please ensure you have:")
    print("   - rice_data.zip in the root directory, OR")
    print("   - rice_data/RiceLeafsDisease folder with train/validation subfolders")
    return False

def check_model():
    """Check if trained model exists"""
    if os.path.exists('rice_disease_model.h5'):
        print("✅ Trained model found: rice_disease_model.h5")
        return True
    else:
        print("❌ Trained model not found!")
        print("You need to train the model first.")
        return False

def train_model():
    """Train the model"""
    print("\n🚀 Starting model training...")
    print("This may take several minutes depending on your dataset size and hardware.")
    
    try:
        result = subprocess.run([sys.executable, 'train_model.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Model training completed successfully!")
            return True
        else:
            print("❌ Model training failed!")
            print("Error output:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return False

def run_app():
    """Run the Flask application"""
    print("\n🌐 Starting Flask web application...")
    print("The application will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\n👋 Application stopped by user")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def main():
    """Main function"""
    print("🍚 Rice Leaf Disease Detection - Quick Start")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check dataset
    if not check_dataset():
        return
    
    # Check if model exists
    if not check_model():
        print("\n📊 Model training is required before running the web app.")
        response = input("Do you want to train the model now? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            if not train_model():
                return
        else:
            print("Please train the model first using: python train_model.py")
            return
    
    # Run the application
    run_app()

if __name__ == "__main__":
    main() 