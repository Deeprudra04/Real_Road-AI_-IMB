#!/usr/bin/env python3
"""
Test script to verify the rice disease detection model works correctly
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json

def test_model_loading():
    """Test if the model can be loaded successfully"""
    print("🔍 Testing model loading...")
    
    try:
        # Load model
        model = tf.keras.models.load_model('rice_disease_model.h5', compile=False)
        print("✅ Model loaded successfully")
        
        # Compile model
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print("✅ Model compiled successfully")
        
        # Print model summary
        print("\n📋 Model Summary:")
        model.summary()
        
        return model
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

def test_disease_classes():
    """Test if disease classes are loaded correctly"""
    print("\n🔍 Testing disease classes...")
    
    try:
        with open('disease_classes.json', 'r') as f:
            classes = json.load(f)
        print(f"✅ Disease classes loaded: {classes}")
        return classes
    except FileNotFoundError:
        print("⚠️  disease_classes.json not found, using default classes")
        classes = [
            'bacterial_leaf_blight',
            'brown_spot', 
            'healthy',
            'leaf_blast',
            'leaf_scald',
            'narrow_brown_spot'
        ]
        print(f"✅ Using default classes: {classes}")
        return classes
    except Exception as e:
        print(f"❌ Error loading disease classes: {e}")
        return None

def test_image_processing():
    """Test image processing with a sample image"""
    print("\n🔍 Testing image processing...")
    
    # Look for a sample image in the dataset
    sample_paths = [
        'rice_data/RiceLeafsDisease/train/healthy/healthy (1).jpg',
        'rice_data/RiceLeafsDisease/validation/healthy/healthy_val (1).jpg',
        'rice_data/RiceLeafsDisease/train/bacterial_leaf_blight/bacterial_leaf_blight (1).JPG'
    ]
    
    sample_image = None
    for path in sample_paths:
        if os.path.exists(path):
            sample_image = path
            break
    
    if sample_image is None:
        print("⚠️  No sample image found for testing")
        return None
    
    try:
        # Load and preprocess image
        img = load_img(sample_image, target_size=(224, 224), color_mode='rgb')
        img_array = img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        print(f"✅ Image processed successfully: {sample_image}")
        print(f"   Shape: {img_array.shape}")
        print(f"   Data type: {img_array.dtype}")
        print(f"   Value range: [{img_array.min():.3f}, {img_array.max():.3f}]")
        
        return img_array
        
    except Exception as e:
        print(f"❌ Error processing image: {e}")
        return None

def test_prediction(model, classes, test_image):
    """Test model prediction"""
    print("\n🔍 Testing model prediction...")
    
    if model is None or test_image is None:
        print("❌ Cannot test prediction - model or test image not available")
        return False
    
    try:
        # Make prediction
        predictions = model.predict(test_image, verbose=0)
        
        # Get results
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        if predicted_class < len(classes):
            disease_name = classes[predicted_class]
        else:
            disease_name = "unknown"
        
        print(f"✅ Prediction successful!")
        print(f"   Predicted disease: {disease_name}")
        print(f"   Confidence: {confidence:.4f}")
        print(f"   Predicted class index: {predicted_class}")
        
        # Show all probabilities
        print("\n📊 All class probabilities:")
        for i, prob in enumerate(predictions[0]):
            if i < len(classes):
                print(f"   {classes[i]}: {prob:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Starting Rice Disease Detection Model Tests")
    print("=" * 50)
    
    # Test 1: Model loading
    model = test_model_loading()
    
    # Test 2: Disease classes
    classes = test_disease_classes()
    
    # Test 3: Image processing
    test_image = test_image_processing()
    
    # Test 4: Prediction
    prediction_success = test_prediction(model, classes, test_image)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary:")
    print(f"   Model loading: {'✅ PASS' if model is not None else '❌ FAIL'}")
    print(f"   Disease classes: {'✅ PASS' if classes is not None else '❌ FAIL'}")
    print(f"   Image processing: {'✅ PASS' if test_image is not None else '❌ FAIL'}")
    print(f"   Prediction: {'✅ PASS' if prediction_success else '❌ FAIL'}")
    
    if all([model is not None, classes is not None, test_image is not None, prediction_success]):
        print("\n🎉 All tests passed! The model is ready to use.")
        print("You can now run: python app.py")
    else:
        print("\n⚠️  Some tests failed. Please check the issues above.")
        print("Make sure you have:")
        print("1. Trained the model: python train_model.py")
        print("2. The rice_disease_model.h5 file exists")
        print("3. The rice_data directory with sample images")

if __name__ == "__main__":
    main() 