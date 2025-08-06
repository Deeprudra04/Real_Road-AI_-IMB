#!/usr/bin/env python3
"""
Rice Leaf Disease Detection - Complete Setup and Run Script
This script handles everything from training to running the web app.
"""

import os
import sys
import subprocess

def check_files():
    """Check if required files exist"""
    model_file = 'rice_disease_model.h5'
    classes_file = 'disease_classes.json'
    
    if os.path.exists(model_file) and os.path.exists(classes_file):
        print("Trained model found!")
        return True
    else:
        print("Trained model not found!")
        return False

def train_model():
    """Train the model"""
    print("\nTraining the model...")
    print("This will take several minutes. Please be patient...")
    
    try:
        result = subprocess.run([sys.executable, 'train_and_save_model.py'], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Model training completed successfully!")
            return True
        else:
            print("Model training failed!")
            print("Error output:")
            print(result.stderr)
            return False
    except Exception as e:
        print(f"Error during training: {e}")
        return False

def run_flask_app():
    """Run the Flask application"""
    print("\nStarting Flask web application...")
    print("The application will be available at: http://localhost:5000")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([sys.executable, 'app.py'])
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"Error running application: {e}")

def main():
    """Main function"""
    print("Rice Leaf Disease Detection - Complete Setup")
    print("=" * 50)
    
    # Check if model already exists
    if check_files():
        print("\nModel is already trained!")
        response = input("Do you want to retrain the model? (y/n): ").lower().strip()
        
        if response in ['y', 'yes']:
            if not train_model():
                return
        else:
            print("Using existing trained model...")
    else:
        # Train the model
        if not train_model():
            return
    
    # Run the Flask app
    run_flask_app()

if __name__ == "__main__":
    main() 