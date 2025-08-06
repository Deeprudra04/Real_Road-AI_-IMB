import os
import random
import zipfile
import tensorflow as tf
from keras.layers import Input, Lambda, Dense, Flatten, GlobalAveragePooling2D, BatchNormalization, Dropout, Activation
from keras.models import Model
from keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split

def extract_dataset():
    """Extract the dataset if it's in zip format"""
    if os.path.exists('rice_data.zip'):
        extracted_dir = "extracted_dataset"
        os.makedirs(extracted_dir, exist_ok=True)
        
        with zipfile.ZipFile('rice_data.zip', 'r') as zip_ref:
            zip_ref.extractall(extracted_dir)
        print(f"Extracted rice_data.zip to {extracted_dir}")
        return extracted_dir
    else:
        print("rice_data.zip not found. Using existing rice_data directory.")
        return "rice_data"

def load_and_preprocess_data(source_path):
    """Load and preprocess the dataset"""
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
    print(f"Training labels shape: {trainy.shape}")
    print(f"Testing labels shape: {testy.shape}")
    
    return trainx, testx, trainy, testy, DISEASE_CLASSES

def create_model():
    """Create the DenseNet121 model"""
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

def train_model():
    """Main training function"""
    print("Starting rice leaf disease detection model training...")
    
    # Extract dataset
    source_path = extract_dataset()
    
    # Load and preprocess data
    print("Loading and preprocessing data...")
    trainx, testx, trainy, testy, disease_classes = load_and_preprocess_data(source_path)
    
    # Create data generator for augmentation
    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        vertical_flip=True,
        rotation_range=30,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.2,
        shear_range=0.2
    )
    
    # Create model
    print("Creating model...")
    model = create_model()
    
    # Compile model
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Train model
    print("Training model...")
    history = model.fit(
        train_datagen.flow(trainx, trainy, batch_size=32),
        validation_data=(testx, testy),
        epochs=10,
        verbose=1
    )
    
    # Save model
    print("Saving model...")
    model.save('rice_disease_model.h5')
    print("Model saved as 'rice_disease_model.h5'")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], 'r', label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], 'b', label='Validation Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], 'r', label='Training Loss')
    plt.plot(history.history['val_loss'], 'b', label='Validation Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_accuracy = model.evaluate(testx, testy, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    return model, disease_classes

if __name__ == "__main__":
    train_model() 