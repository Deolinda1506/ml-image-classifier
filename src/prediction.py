import os
import numpy as np
import cv2
import tensorflow as tf
from PIL import Image
import json
from datetime import datetime
import time
import io

class ImagePredictor:
    def __init__(self, model_path='models/image_classifier.h5'):
        """
        Initialize the image predictor
        
        Args:
            model_path (str): Path to the trained model
        """
        self.model = None
        self.class_names = ['cat', 'dog']
        self.img_size = (224, 224)
        self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load the trained model
        
        Args:
            model_path (str): Path to the model file
        """
        try:
            self.model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
            
            # Load metadata if available
            metadata_path = model_path.replace('.h5', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.img_size = tuple(metadata['img_size'])
                    self.class_names = metadata['class_names']
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for prediction
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image from {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, self.img_size)
        
        # Normalize pixel values
        img = img.astype(np.float32) / 255.0
        
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        
        return img
    
    def predict(self, image_path, return_probabilities=False):
        """
        Make prediction on a single image
        
        Args:
            image_path (str): Path to the image file
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            dict: Prediction results
        """
        start_time = time.time()
        
        try:
            # Preprocess image
            img = self.preprocess_image(image_path)
            
            # Make prediction
            predictions = self.model.predict(img, verbose=0)
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx])
            
            # Calculate prediction time
            prediction_time = time.time() - start_time
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'prediction_time': prediction_time,
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.class_names, predictions[0])
                }
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'prediction_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat(),
                'image_path': image_path
            }
    
    def predict_batch(self, image_paths, return_probabilities=False):
        """
        Make predictions on multiple images
        
        Args:
            image_paths (list): List of image file paths
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            list: List of prediction results
        """
        results = []
        
        for image_path in image_paths:
            result = self.predict(image_path, return_probabilities)
            results.append(result)
        
        return results
    
    def predict_from_bytes(self, image_bytes, return_probabilities=False):
        """
        Make prediction from image bytes (for API usage)
        
        Args:
            image_bytes (bytes): Image data as bytes
            return_probabilities (bool): Whether to return class probabilities
            
        Returns:
            dict: Prediction results
        """
        start_time = time.time()
        
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to numpy array
            img = np.array(image)
            
            # Convert RGBA to RGB if necessary
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            
            # Resize image
            img = cv2.resize(img, self.img_size)
            
            # Normalize pixel values
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            # Make prediction
            predictions = self.model.predict(img, verbose=0)
            
            # Get predicted class and confidence
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_names[predicted_class_idx]
            confidence = float(predictions[0][predicted_class_idx])
            
            # Calculate prediction time
            prediction_time = time.time() - start_time
            
            result = {
                'predicted_class': predicted_class,
                'confidence': confidence,
                'prediction_time': prediction_time,
                'timestamp': datetime.now().isoformat()
            }
            
            if return_probabilities:
                result['probabilities'] = {
                    class_name: float(prob) 
                    for class_name, prob in zip(self.class_names, predictions[0])
                }
            
            return result
            
        except Exception as e:
            return {
                'error': str(e),
                'prediction_time': time.time() - start_time,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_model_info(self):
        """
        Get information about the loaded model
        
        Returns:
            dict: Model information
        """
        if self.model is None:
            return {'error': 'No model loaded'}
        
        return {
            'model_type': 'CNN Image Classifier',
            'input_shape': self.model.input_shape,
            'output_shape': self.model.output_shape,
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'total_params': self.model.count_params(),
            'trainable_params': sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        }
    
    def validate_image(self, image_path):
        """
        Validate if an image can be processed
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Validation result
        """
        try:
            # Check if file exists
            if not os.path.exists(image_path):
                return {'valid': False, 'error': 'File does not exist'}
            
            # Check file size
            file_size = os.path.getsize(image_path)
            if file_size == 0:
                return {'valid': False, 'error': 'File is empty'}
            
            # Try to load image
            img = cv2.imread(image_path)
            if img is None:
                return {'valid': False, 'error': 'Could not load image'}
            
            # Check image dimensions
            height, width, channels = img.shape
            if channels != 3:
                return {'valid': False, 'error': 'Image must have 3 channels (RGB)'}
            
            return {
                'valid': True,
                'file_size': file_size,
                'dimensions': {'width': width, 'height': height, 'channels': channels}
            }
            
        except Exception as e:
            return {'valid': False, 'error': str(e)} 