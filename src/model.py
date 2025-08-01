import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import cv2
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ImageClassifier:
    def __init__(self, model_type='cnn', img_size=(128, 128), num_classes=2):
        """
        Initialize the image classifier with TensorFlow/Keras CNN
        
        Args:
            model_type (str): Type of model to use ('cnn', 'custom_cnn')
            img_size (tuple): Input image size (width, height)
            num_classes (int): Number of classes
        """
        self.model_type = model_type
        self.img_size = img_size
        self.num_classes = num_classes
        self.model = None
        self.class_names = ['cat', 'dog']
        self.training_history = None
        
        # Data augmentation parameters
        self.augmentation_params = {
            'rotation_range': 20,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'horizontal_flip': True,
            'brightness_range': [0.8, 1.2],
            'zoom_range': 0.1,
            'fill_mode': 'nearest'
        }
        
    def build_model(self):
        """
        Build CNN model with L2 regularization, dropout, and advanced features
        """
        if self.model_type == 'cnn':
            model = models.Sequential([
                layers.InputLayer(input_shape=(*self.img_size, 3)),
                
                # Convolutional Layer 1 with L2 Regularization
                layers.Conv2D(32, (3, 3), activation='relu', 
                             kernel_regularizer=regularizers.l2(0.01), 
                             padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                # Convolutional Layer 2
                layers.Conv2D(64, (3, 3), activation='relu', 
                             kernel_regularizer=regularizers.l2(0.01), 
                             padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                # Convolutional Layer 3
                layers.Conv2D(128, (3, 3), activation='relu', 
                             kernel_regularizer=regularizers.l2(0.01), 
                             padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                # Convolutional Layer 4
                layers.Conv2D(256, (3, 3), activation='relu', 
                             kernel_regularizer=regularizers.l2(0.01), 
                             padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                
                # Flatten and Dense Layers
                layers.Flatten(),
                layers.Dense(512, activation='relu', 
                            kernel_regularizer=regularizers.l2(0.01)),
                layers.BatchNormalization(),
                layers.Dropout(0.5),  # Dropout for regularization
                
                layers.Dense(256, activation='relu', 
                            kernel_regularizer=regularizers.l2(0.01)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                layers.Dense(128, activation='relu', 
                            kernel_regularizer=regularizers.l2(0.01)),
                layers.BatchNormalization(),
                layers.Dropout(0.2),
                
                layers.Dense(self.num_classes, activation='softmax')  # Output layer
            ])
        else:
            # Custom CNN architecture
            model = models.Sequential([
                layers.InputLayer(input_shape=(*self.img_size, 3)),
                
                # First Convolutional Block
                layers.Conv2D(32, (3, 3), activation='relu', 
                             kernel_regularizer=regularizers.l2(0.01), 
                             padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(32, (3, 3), activation='relu', 
                             kernel_regularizer=regularizers.l2(0.01), 
                             padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Second Convolutional Block
                layers.Conv2D(64, (3, 3), activation='relu', 
                             kernel_regularizer=regularizers.l2(0.01), 
                             padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(64, (3, 3), activation='relu', 
                             kernel_regularizer=regularizers.l2(0.01), 
                             padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Third Convolutional Block
                layers.Conv2D(128, (3, 3), activation='relu', 
                             kernel_regularizer=regularizers.l2(0.01), 
                             padding='same'),
                layers.BatchNormalization(),
                layers.Conv2D(128, (3, 3), activation='relu', 
                             kernel_regularizer=regularizers.l2(0.01), 
                             padding='same'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Dense Layers
                layers.Flatten(),
                layers.Dense(512, activation='relu', 
                            kernel_regularizer=regularizers.l2(0.01)),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                
                layers.Dense(256, activation='relu', 
                            kernel_regularizer=regularizers.l2(0.01)),
                layers.BatchNormalization(),
                layers.Dropout(0.3),
                
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        # Compile the model with Adam optimizer and categorical cross-entropy loss
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall']
        )
        
        self.model = model
        return model
    
    def create_data_generators(self, X_train, y_train, X_val, y_val, batch_size=32):
        """
        Create data generators with augmentation for training and validation
        """
        # Convert labels to categorical
        y_train_cat = keras.utils.to_categorical(y_train, self.num_classes)
        y_val_cat = keras.utils.to_categorical(y_val, self.num_classes)
        
        # Training data generator with augmentation
        train_datagen = ImageDataGenerator(
            **self.augmentation_params,
            rescale=1./255  # Normalize pixel values
        )
        
        # Validation data generator (no augmentation, just rescaling)
        val_datagen = ImageDataGenerator(
            rescale=1./255
        )
        
        # Create generators
        train_generator = train_datagen.flow(
            X_train, y_train_cat, 
            batch_size=batch_size
        )
        
        val_generator = val_datagen.flow(
            X_val, y_val_cat, 
            batch_size=batch_size
        )
        
        return train_generator, val_generator, y_train_cat, y_val_cat
    
    def train(self, X_train, y_train, X_val=None, y_val=None, 
              epochs=50, batch_size=32, model_save_path=None):
        """
        Train the model with advanced callbacks and monitoring
        """
        print(f"Training TensorFlow CNN model with {epochs} epochs...")
        
        # Split validation data if not provided
        if X_val is None or y_val is None:
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
        else:
            X_train_split, y_train_split = X_train, y_train
        
        # Create data generators
        train_generator, val_generator, y_train_cat, y_val_cat = self.create_data_generators(
            X_train_split, y_train_split, X_val, y_val, batch_size
        )
        
        # Define callbacks
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(
                monitor='val_loss', 
                patience=5, 
                restore_best_weights=True,
                verbose=1
            ),
            
            # Model checkpoint to save best model
            ModelCheckpoint(
                filepath=model_save_path or 'best_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Reduce learning rate when plateau is reached
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]
        
        # Calculate steps per epoch
        steps_per_epoch = len(X_train_split) // batch_size
        validation_steps = len(X_val) // batch_size
        
        # Train the model
        start_time = datetime.now()
        
        history = self.model.fit(
            train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            validation_steps=validation_steps,
            callbacks=callbacks,
            verbose=1
        )
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        self.training_history = history
        
        print(f"\nTraining completed in {training_duration:.2f} seconds")
        print(f"Training completed in {training_duration/60:.2f} minutes")
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model with comprehensive metrics
        """
        print("Evaluating TensorFlow CNN model...")
        
        # Preprocess test data
        X_test_normalized = X_test / 255.0
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test_normalized, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Convert to categorical for metrics
        y_test_cat = keras.utils.to_categorical(y_test, self.num_classes)
        
        # Calculate comprehensive metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # ROC-AUC for binary classification
        if self.num_classes == 2:
            roc_auc = np.mean([np.trapz(y_test_cat[:, i], y_pred_proba[:, i]) 
                              for i in range(self.num_classes)])
        else:
            roc_auc = np.mean([np.trapz(y_test_cat[:, i], y_pred_proba[:, i]) 
                              for i in range(self.num_classes)])
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Classification report
        class_report = classification_report(y_test, y_pred, target_names=self.class_names)
        
        # Calculate per-class metrics
        precision_per_class = precision_score(y_test, y_pred, average=None)
        recall_per_class = recall_score(y_test, y_pred, average=None)
        f1_per_class = f1_score(y_test, y_pred, average=None)
        
        # Store results
        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'classification_report': class_report,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba,
            'precision_per_class': precision_per_class,
            'recall_per_class': recall_per_class,
            'f1_per_class': f1_per_class
        }
        
        # Print results
        print("\n" + "="*50)
        print("TENSORFLOW CNN MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        print("\nPer-Class Metrics:")
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name}: Precision={precision_per_class[i]:.4f}, "
                  f"Recall={recall_per_class[i]:.4f}, F1={f1_per_class[i]:.4f}")
        
        print("\nClassification Report:")
        print(class_report)
        
        return results
    
    def predict(self, image):
        """
        Make prediction on a single image or batch of images
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Preprocess image
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        image_normalized = image / 255.0
        
        # Make prediction
        prediction_proba = self.model.predict(image_normalized, verbose=0)
        prediction = np.argmax(prediction_proba, axis=1)
        
        return prediction, prediction_proba
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        if self.training_history is None:
            print("No training history available!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.training_history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.training_history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.training_history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.training_history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Precision
        axes[1, 0].plot(self.training_history.history['Precision'], label='Training Precision')
        axes[1, 0].plot(self.training_history.history['val_Precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Recall
        axes[1, 1].plot(self.training_history.history['Recall'], label='Training Recall')
        axes[1, 1].plot(self.training_history.history['val_Recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model"""
        if self.model is not None:
            self.model.save(filepath)
            print(f"Model saved to {filepath}")
        else:
            print("No model to save!")
    
    def load_model(self, filepath):
        """Load a trained model"""
        self.model = keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def get_model_info(self):
        """Get model information"""
        info = {
            'model_type': 'TensorFlow CNN',
            'img_size': self.img_size,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'architecture': 'Sequential CNN with L2 regularization and dropout'
        }
        return info 