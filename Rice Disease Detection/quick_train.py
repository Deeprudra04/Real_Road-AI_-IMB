#!/usr/bin/env python3
"""
Quick Train and Save Model
This script trains the model with fewer epochs and ensures it saves properly.
"""

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import json

def load_data():
    """Load and preprocess the dataset"""
    source_path = "rice_data"
    TRAINING_DIR = os.path.join(source_path, 'RiceLeafsDisease/train')
    VALIDATION_DIR = os.path.join(source_path, 'RiceLeafsDisease/validation')
    
    # Disease classes
    DISEASE_CLASSES = [
        'bacterial_leaf_blight',
        'brown_spot', 
        'healthy',
        'leaf_blast',
        'leaf_scald',
        'narrow_brown_spot'
    ]
    
    dataset = []
    shape = (224, 224)
    count = 0
    
    print("Loading training data...")
    # Load training data
    for disease_class in DISEASE_CLASSES:
        path = os.path.join(TRAINING_DIR, disease_class)
        if os.path.exists(path):
            for k in os.listdir(path):
                try:
                    image = load_img(os.path.join(path, k), target_size=shape, color_mode='rgb')
                    image = img_to_array(image)
                    image = image / 255.0
                    dataset += [[image, count]]
                except Exception as e:
                    print(f"Error loading image {k}: {e}")
            count += 1
    
    # Load validation data
    testset = []
    count = 0
    print("Loading validation data...")
    for disease_class in DISEASE_CLASSES:
        path = os.path.join(VALIDATION_DIR, disease_class)
        if os.path.exists(path):
            for im in os.listdir(path):
                try:
                    image = load_img(os.path.join(path, im), target_size=shape, color_mode='rgb')
                    image = img_to_array(image)
                    image = image / 255.0
                    testset += [[image, count]]
                except Exception as e:
                    print(f"Error loading validation image {im}: {e}")
            count += 1
    
    # Separate data and labels
    data, trainlabels = zip(*dataset)
    test, testlabels = zip(*testset)
    
    # Convert to categorical
    labels1 = to_categorical(trainlabels)
    labels = np.array(labels1)
    
    # Convert to numpy arrays
    data = np.array(data)
    test = np.array(test)
    
    # Split data
    trainx, testx, trainy, testy = train_test_split(data, labels, test_size=0.2, random_state=42)
    
    print(f"Training data shape: {trainx.shape}")
    print(f"Testing data shape: {testx.shape}")
    
    return trainx, testx, trainy, testy, DISEASE_CLASSES

def create_model():
    """Create the DenseNet121 model"""
    print("Creating DenseNet121 model...")
    pretrained_model = tf.keras.applications.DenseNet121(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    
    inputs = pretrained_model.input
    x = tf.keras.layers.Flatten()(pretrained_model.output)
    x = tf.keras.layers.Dense(1024, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(6, activation='softmax')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model

def train_and_save():
    """Train the model and save it"""
    print("Starting Quick Training...")
    print("=" * 40)
    
    # Load data
    trainx, testx, trainy, testy, disease_classes = load_data()
    
    # Create model
    model = create_model()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model (fewer epochs for faster completion)
    print("Training model (5 epochs)...")
    history = model.fit(
        trainx, trainy,
        validation_data=(testx, testy),
        epochs=5,
        batch_size=32,
        verbose=1
    )
    
    # Save model
    print("Saving model...")
    model.save('rice_disease_model.h5')
    print("Model saved as 'rice_disease_model.h5'")
    
    # Save disease classes
    with open('disease_classes.json', 'w') as f:
        json.dump(disease_classes, f)
    print("Disease classes saved as 'disease_classes.json'")
    
    # Evaluate model
    test_loss, test_accuracy = model.evaluate(testx, testy, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    print("\nTraining completed successfully!")
    print("Files created:")
    print("   - rice_disease_model.h5")
    print("   - disease_classes.json")
    
    return model, disease_classes

if __name__ == "__main__":
    train_and_save() 