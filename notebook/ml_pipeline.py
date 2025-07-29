#!/usr/bin/env python3
"""
ML Image Classification Pipeline - Python Script Version

This script demonstrates the complete machine learning pipeline for image classification of cats and dogs.
You can run this as a Python script or convert it to a Jupyter notebook.

To convert to notebook:
jupyter nbconvert --to notebook --execute ml_pipeline.py --output ml_pipeline.ipynb
"""

# Import required libraries
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append('../src')

# Import our custom modules
from src.preprocessing import ImagePreprocessor
from src.model import ImageClassifier
from src.prediction import ImagePredictor

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# ============================================================================
# 1. DATA PREPROCESSING AND ANALYSIS
# ============================================================================

print("\n" + "="*50)
print("1. DATA PREPROCESSING AND ANALYSIS")
print("="*50)

# Initialize preprocessor
preprocessor = ImagePreprocessor()

# Analyze dataset
print("Analyzing dataset...")
dataset_stats = preprocessor.analyze_dataset('../data/train')
print("Dataset statistics:")
print(f"Total images: {dataset_stats['total_images']}")
print(f"Class distribution: {dataset_stats['class_counts']}")

# Load and preprocess dataset
print("\nLoading dataset...")
images, labels, file_paths = preprocessor.load_dataset('../data/train')

print(f"Loaded {len(images)} images")
print(f"Image shape: {images.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Unique labels: {np.unique(labels)}")

# Create data generators with augmentation
print("\nCreating data generators...")
train_generator, validation_generator = preprocessor.create_data_generators('../data/train', validation_split=0.2)

print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {validation_generator.samples}")
print(f"Number of classes: {train_generator.num_classes}")
print(f"Class indices: {train_generator.class_indices}")

# ============================================================================
# 2. MODEL CREATION AND TRAINING
# ============================================================================

print("\n" + "="*50)
print("2. MODEL CREATION AND TRAINING")
print("="*50)

# Initialize classifier
classifier = ImageClassifier(img_size=(224, 224), num_classes=2, learning_rate=0.001)

# Build model with pre-trained MobileNetV2
print("Building model...")
model = classifier.build_model(use_pretrained=True)

# Display model summary
print("Model Summary:")
model.summary()

# Train the model
print("\nStarting model training...")
history = classifier.train(train_generator, validation_generator, epochs=30, batch_size=32)

print("Training completed!")

# Plot training history
classifier.plot_training_history('training_history.png')
print("Training history plot saved as 'training_history.png'")

# ============================================================================
# 3. MODEL EVALUATION
# ============================================================================

print("\n" + "="*50)
print("3. MODEL EVALUATION")
print("="*50)

# Evaluate model on validation set
print("Evaluating model...")
metrics = classifier.evaluate_model(validation_generator)

print("\nModel Performance Metrics:")
print(f"Test Loss: {metrics['test_loss']:.4f}")
print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
print(f"Test Precision: {metrics['test_precision']:.4f}")
print(f"Test Recall: {metrics['test_recall']:.4f}")
print(f"Test F1 Score: {metrics['test_f1']:.4f}")

# Display classification report
print("\nClassification Report:")
classification_rep = metrics['classification_report']
for class_name in ['cat', 'dog']:
    print(f"\n{class_name.upper()}:")
    print(f"  Precision: {classification_rep[class_name]['precision']:.4f}")
    print(f"  Recall: {classification_rep[class_name]['recall']:.4f}")
    print(f"  F1-Score: {classification_rep[class_name]['f1-score']:.4f}")
    print(f"  Support: {classification_rep[class_name]['support']}")

# Plot confusion matrix
conf_matrix = np.array(metrics['confusion_matrix'])

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Cat', 'Dog'], yticklabels=['Cat', 'Dog'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ============================================================================
# 4. MODEL TESTING AND PREDICTION
# ============================================================================

print("\n" + "="*50)
print("4. MODEL TESTING AND PREDICTION")
print("="*50)

# Save the trained model
print("Saving model...")
classifier.save_model('../models/image_classifier.h5')
print("Model saved successfully!")

# Test prediction on sample images
predictor = ImagePredictor('../models/image_classifier.h5')

# Test on a few sample images
test_images = ['../data/test/0.jpg', '../data/test/1.jpg', '../data/test/2.jpg']

print("Testing predictions on sample images:")
for img_path in test_images:
    if os.path.exists(img_path):
        result = predictor.predict(img_path, return_probabilities=True)
        print(f"\nImage: {os.path.basename(img_path)}")
        print(f"Predicted: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Prediction time: {result['prediction_time']:.4f}s")
        if 'probabilities' in result:
            print("Probabilities:")
            for class_name, prob in result['probabilities'].items():
                print(f"  {class_name}: {prob:.4f}")

# ============================================================================
# 5. FEATURE ANALYSIS AND INTERPRETATIONS
# ============================================================================

print("\n" + "="*50)
print("5. FEATURE ANALYSIS AND INTERPRETATIONS")
print("="*50)

# Analyze model features using Grad-CAM
def generate_gradcam(model, img_array, layer_name='global_average_pooling2d'):
    """Generate Grad-CAM visualization"""
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(layer_name).input, model.output]
    )
    
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        loss = predictions[:, tf.argmax(predictions[0])]
    
    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# Test Grad-CAM on sample images
sample_img_path = '../data/test/0.jpg'
if os.path.exists(sample_img_path):
    # Load and preprocess image
    img = cv2.imread(sample_img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (224, 224))
    img_array = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)
    
    # Generate heatmap
    heatmap = generate_gradcam(classifier.model, img_array)
    
    # Resize heatmap to match original image
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(img)
    axes[2].imshow(heatmap, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('gradcam_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================================================
# 6. MODEL PERFORMANCE ANALYSIS
# ============================================================================

print("\n" + "="*50)
print("6. MODEL PERFORMANCE ANALYSIS")
print("="*50)

# Analyze model performance across different metrics
performance_metrics = {
    'Accuracy': metrics['test_accuracy'],
    'Precision': metrics['test_precision'],
    'Recall': metrics['test_recall'],
    'F1-Score': metrics['test_f1']
}

# Create performance visualization
plt.figure(figsize=(10, 6))
metrics_names = list(performance_metrics.keys())
metrics_values = list(performance_metrics.values())

bars = plt.bar(metrics_names, metrics_values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
plt.title('Model Performance Metrics', fontsize=16, fontweight='bold')
plt.ylabel('Score', fontsize=12)
plt.ylim(0, 1)

# Add value labels on bars
for bar, value in zip(bars, metrics_values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('performance_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nModel Performance Summary:")
for metric, value in performance_metrics.items():
    print(f"{metric}: {value:.4f}")

# ============================================================================
# 7. MODEL OPTIMIZATION INSIGHTS
# ============================================================================

print("\n" + "="*50)
print("7. MODEL OPTIMIZATION INSIGHTS")
print("="*50)

# Analyze training history for optimization insights
if history:
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training vs Validation Accuracy
    axes[0, 0].plot(epochs, history.history['accuracy'], 'b-', label='Training Accuracy')
    axes[0, 0].plot(epochs, history.history['val_accuracy'], 'r-', label='Validation Accuracy')
    axes[0, 0].set_title('Training vs Validation Accuracy')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Training vs Validation Loss
    axes[0, 1].plot(epochs, history.history['loss'], 'b-', label='Training Loss')
    axes[0, 1].plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss')
    axes[0, 1].set_title('Training vs Validation Loss')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision
    axes[1, 0].plot(epochs, history.history['precision'], 'g-', label='Training Precision')
    axes[1, 0].plot(epochs, history.history['val_precision'], 'm-', label='Validation Precision')
    axes[1, 0].set_title('Training vs Validation Precision')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Precision')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Recall
    axes[1, 1].plot(epochs, history.history['recall'], 'c-', label='Training Recall')
    axes[1, 1].plot(epochs, history.history['val_recall'], 'y-', label='Validation Recall')
    axes[1, 1].set_title('Training vs Validation Recall')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Recall')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_insights.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nModel Complexity:")
    print(f"Total parameters: {classifier.model.count_params():,}")
    print(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in classifier.model.trainable_weights]):,}")

# ============================================================================
# 8. CONCLUSION AND NEXT STEPS
# ============================================================================

print("\n" + "="*50)
print("8. CONCLUSION AND NEXT STEPS")
print("="*50)

print("=== ML Pipeline Summary ===\n")
print(f"Dataset Size: {dataset_stats['total_images']} images")
print(f"Classes: {list(dataset_stats['class_counts'].keys())}")
print(f"Model Architecture: MobileNetV2 + Custom Classifier")
print(f"Final Test Accuracy: {metrics['test_accuracy']:.4f}")
print(f"Final Test F1-Score: {metrics['test_f1']:.4f}")
print(f"\nModel saved to: ../models/image_classifier.h5")
print("\nGenerated Files:")
print("- training_history.png")
print("- confusion_matrix.png")
print("- gradcam_analysis.png")
print("- performance_metrics.png")
print("- training_insights.png")

print("\nNext Steps:")
print("1. Deploy the model using the Flask API")
print("2. Set up monitoring and logging")
print("3. Implement retraining pipeline")
print("4. Create web interface for predictions")

print("\n🎉 ML Pipeline completed successfully!") 