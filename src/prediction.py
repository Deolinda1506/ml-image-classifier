import os
import numpy as np
import sqlite3
from datetime import datetime
import joblib
from preprocessing import ImagePreprocessor
import warnings
warnings.filterwarnings('ignore')

# Import TensorFlow
import tensorflow as tf
from tensorflow import keras

class PredictionService:
    def __init__(self, model_path=None):
        """
        Initialize the prediction service for TensorFlow CNN model
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model = None
        self.class_names = ['cat', 'dog']
        self.preprocessor = ImagePreprocessor(img_size=(128, 128), batch_size=32)
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """Load TensorFlow CNN model"""
        try:
            # Load the model
            self.model = keras.models.load_model(model_path)
            print(f"TensorFlow CNN model loaded from {model_path}")
        except Exception as e:
            print(f"Error loading TensorFlow model: {e}")
            raise
    
    def predict_uploaded_image(self, image_file):
        """
        Make prediction on an uploaded image file
        
        Args:
            image_file: Uploaded file object
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            raise ValueError("No model loaded!")
        
        try:
            # Preprocess uploaded image
            image = self.preprocessor.preprocess_uploaded_image(image_file)
            
            # Make prediction
            prediction, probability = self._predict_single_image(image)
            
            # Get class name
            predicted_class = self.class_names[prediction[0]]
            confidence = np.max(probability[0])
            
            # Log prediction
            self._log_prediction("uploaded_file", predicted_class, confidence)
            
            return {
                'prediction': predicted_class,
                'confidence': float(confidence),
                'probabilities': {
                    'cat': float(probability[0][0]),
                    'dog': float(probability[0][1])
                }
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            raise
    
    def _predict_single_image(self, image):
        """
        Make prediction on a single image
        
        Args:
            image: Preprocessed image array
            
        Returns:
            tuple: (prediction, probability)
        """
        if self.model is None:
            raise ValueError("No model loaded!")
        
        # Ensure image is in the right format for TensorFlow
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
        
        # Normalize image (if not already done)
        if image.max() > 1.0:
            image = image / 255.0
        
        # Make prediction
        prediction_proba = self.model.predict(image, verbose=0)
        prediction = np.argmax(prediction_proba, axis=1)
        
        return prediction, prediction_proba
    
    def _log_prediction(self, image_name, predicted_class, confidence):
        """Log prediction to database"""
        try:
            conn = sqlite3.connect('predictions.db')
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO predictions (image_name, predicted_class, confidence)
                VALUES (?, ?, ?)
            ''', (image_name, predicted_class, confidence))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Error logging prediction: {e}")
    
    def get_model_info(self):
        """Get model information"""
        if self.model is None:
            return None
        
        return {
            'model_type': 'TensorFlow CNN',
            'class_names': self.class_names,
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'total_params': self.model.count_params()
        } 