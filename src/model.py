import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import pickle
import json
from datetime import datetime

class ImageClassifier:
    def __init__(self, img_size=(224, 224), num_classes=2, learning_rate=0.001):
        """
        Initialize the image classifier
        
        Args:
            img_size (tuple): Input image size
            num_classes (int): Number of classes
            learning_rate (float): Learning rate for optimizer
        """
        self.img_size = img_size
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.model = None
        self.history = None
        self.class_names = ['cat', 'dog']
        
    def build_model(self, use_pretrained=True):
        """
        Build the CNN model
        
        Args:
            use_pretrained (bool): Whether to use pre-trained MobileNetV2
            
        Returns:
            tensorflow.keras.Model: Compiled model
        """
        if use_pretrained:
            # Use pre-trained MobileNetV2 as base
            base_model = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_size[0], self.img_size[1], 3)
            )
            
            # Freeze the base model layers
            base_model.trainable = False
            
            # Create the model
            self.model = models.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dropout(0.2),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(256, activation='relu'),
                layers.Dropout(0.2),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        else:
            # Custom CNN architecture
            self.model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size[0], self.img_size[1], 3)),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.Dropout(0.3),
                layers.Dense(self.num_classes, activation='softmax')
            ])
        
        # Compile the model
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return self.model
    
    def train(self, train_generator, validation_generator, epochs=50, batch_size=32):
        """
        Train the model
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            
        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
        
        model_checkpoint = callbacks.ModelCheckpoint(
            'models/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            epochs=epochs,
            validation_data=validation_generator,
            callbacks=[early_stopping, reduce_lr, model_checkpoint],
            verbose=1
        )
        
        return self.history
    
    def evaluate_model(self, test_generator):
        """
        Evaluate the model on test data
        
        Args:
            test_generator: Test data generator
            
        Returns:
            dict: Evaluation metrics
        """
        # Evaluate the model
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(test_generator, verbose=0)
        
        # Calculate F1 score
        test_f1 = 2 * (test_precision * test_recall) / (test_precision + test_recall)
        
        # Get predictions
        predictions = self.model.predict(test_generator)
        y_pred = np.argmax(predictions, axis=1)
        
        # Get true labels - handle different generator types
        if hasattr(test_generator, 'classes'):
            # Standard Keras generator
            y_true = test_generator.classes
        else:
            # Custom generator - extract labels from the generator
            y_true = []
            test_generator.reset()
            for i in range(len(test_generator)):
                _, batch_labels = test_generator[i]
                y_true.extend(np.argmax(batch_labels, axis=1))
            y_true = np.array(y_true)
        
        # Calculate additional metrics
        from sklearn.metrics import classification_report, confusion_matrix
        
        classification_rep = classification_report(y_true, y_pred, target_names=self.class_names, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'test_loss': test_loss,
            'test_accuracy': test_accuracy,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'test_f1': test_f1,
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist()
        }
        
        return metrics
    
    def plot_training_history(self, save_path='training_history.png'):
        """
        Plot training history
        
        Args:
            save_path (str): Path to save the plot
        """
        if self.history is None:
            print("No training history available. Train the model first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Training Accuracy')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Training Loss')
        axes[0, 1].plot(self.history.history['val_loss'], label='Validation Loss')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Precision
        axes[1, 0].plot(self.history.history['precision'], label='Training Precision')
        axes[1, 0].plot(self.history.history['val_precision'], label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Recall
        axes[1, 1].plot(self.history.history['recall'], label='Training Recall')
        axes[1, 1].plot(self.history.history['val_recall'], label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_model(self, model_path='models/image_classifier.h5'):
        """
        Save the trained model
        
        Args:
            model_path (str): Path to save the model
        """
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        
        # Save model metadata
        metadata = {
            'img_size': self.img_size,
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'learning_rate': self.learning_rate,
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = model_path.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self, model_path='models/image_classifier.h5'):
        """
        Load a trained model
        
        Args:
            model_path (str): Path to the model file
        """
        self.model = models.load_model(model_path)
        
        # Load metadata
        metadata_path = model_path.replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                self.img_size = tuple(metadata['img_size'])
                self.num_classes = metadata['num_classes']
                self.class_names = metadata['class_names']
                self.learning_rate = metadata['learning_rate']
    
    def retrain_model(self, new_data_generator, epochs=10):
        """
        Retrain the model with new data
        
        Args:
            new_data_generator: New data generator
            epochs (int): Number of training epochs
            
        Returns:
            tensorflow.keras.callbacks.History: Training history
        """
        if self.model is None:
            raise ValueError("No model loaded. Please load a model first.")
        
        # Unfreeze some layers for fine-tuning
        for layer in self.model.layers[-10:]:
            layer.trainable = True
        
        # Recompile with lower learning rate for fine-tuning
        self.model.compile(
            optimizer=optimizers.Adam(learning_rate=self.learning_rate * 0.1),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        # Callbacks for retraining
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        # Retrain
        retrain_history = self.model.fit(
            new_data_generator,
            epochs=epochs,
            callbacks=[early_stopping],
            verbose=1
        )
        
        return retrain_history 