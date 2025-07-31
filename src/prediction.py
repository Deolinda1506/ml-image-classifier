import os
import numpy as np
import sqlite3
from datetime import datetime
import joblib
from preprocessing import ImagePreprocessor
import warnings
warnings.filterwarnings('ignore')

class PredictionService:
    def __init__(self, model_path=None):
        """
        Initialize the prediction service
        
        Args:
            model_path (str): Path to the trained model file
        """
        self.model = None
        self.scaler = None
        self.class_names = ['cat', 'dog']
        self.preprocessor = ImagePreprocessor()
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
    
    def load_model(self, model_path):
        """
        Load a trained model
        
        Args:
            model_path (str): Path to the model file
        """
        try:
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.class_names = model_data['class_names']
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict_image(self, image_path):
        """
        Make prediction on an image file
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            dict: Prediction results
        """
        if self.model is None:
            raise ValueError("No model loaded!")
        
        try:
            # Load and preprocess image
            image = self.preprocessor.load_and_preprocess_image(image_path)
            
            # Make prediction
            prediction, probability = self.model.predict(image)
            
            # Get class name
            predicted_class = self.class_names[prediction]
            confidence = np.max(probability)
            
            # Log prediction
            self._log_prediction(image_path, predicted_class, confidence)
            
            return {
                'prediction': predicted_class,
                'confidence': float(confidence),
                'probabilities': {
                    'cat': float(probability[0]),
                    'dog': float(probability[1])
                },
                'image_path': image_path
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
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
            prediction, probability = self.model.predict(image)
            
            # Get class name
            predicted_class = self.class_names[prediction]
            confidence = np.max(probability)
            
            # Log prediction
            self._log_prediction("uploaded_file", predicted_class, confidence)
            
            return {
                'prediction': predicted_class,
                'confidence': float(confidence),
                'probabilities': {
                    'cat': float(probability[0]),
                    'dog': float(probability[1])
                }
            }
            
        except Exception as e:
            print(f"Error making prediction: {e}")
            raise
    
    def predict_batch(self, image_paths):
        """
        Make predictions on multiple images
        
        Args:
            image_paths (list): List of image file paths
            
        Returns:
            list: List of prediction results
        """
        if self.model is None:
            raise ValueError("No model loaded!")
        
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict_image(image_path)
                results.append(result)
            except Exception as e:
                print(f"Error predicting {image_path}: {e}")
                results.append({
                    'prediction': 'error',
                    'confidence': 0.0,
                    'error': str(e),
                    'image_path': image_path
                })
        
        return results
    
    def _log_prediction(self, image_path, prediction, confidence):
        """
        Log prediction to database
        
        Args:
            image_path (str): Path to the image
            prediction (str): Predicted class
            confidence (float): Prediction confidence
        """
        try:
            conn = sqlite3.connect('predictions.db')
            cursor = conn.cursor()
            
            # Create table if not exists
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    image_path TEXT,
                    prediction TEXT,
                    confidence REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Insert prediction
            cursor.execute('''
                INSERT INTO predictions (image_path, prediction, confidence)
                VALUES (?, ?, ?)
            ''', (image_path, prediction, confidence))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error logging prediction: {e}")
    
    def get_prediction_history(self, limit=100):
        """
        Get prediction history from database
        
        Args:
            limit (int): Maximum number of records to return
            
        Returns:
            list: List of prediction records
        """
        try:
            conn = sqlite3.connect('predictions.db')
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT image_path, prediction, confidence, timestamp
                FROM predictions
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            records = cursor.fetchall()
            conn.close()
            
            return [
                {
                    'image_path': record[0],
                    'prediction': record[1],
                    'confidence': record[2],
                    'timestamp': record[3]
                }
                for record in records
            ]
            
        except Exception as e:
            print(f"Error getting prediction history: {e}")
            return []
    
    def get_prediction_stats(self):
        """
        Get prediction statistics
        
        Returns:
            dict: Prediction statistics
        """
        try:
            conn = sqlite3.connect('predictions.db')
            cursor = conn.cursor()
            
            # Total predictions
            cursor.execute('SELECT COUNT(*) FROM predictions')
            total_predictions = cursor.fetchone()[0]
            
            # Predictions by class
            cursor.execute('''
                SELECT prediction, COUNT(*) 
                FROM predictions 
                GROUP BY prediction
            ''')
            class_counts = dict(cursor.fetchall())
            
            # Average confidence
            cursor.execute('SELECT AVG(confidence) FROM predictions')
            avg_confidence = cursor.fetchone()[0] or 0.0
            
            # Recent predictions (last 24 hours)
            cursor.execute('''
                SELECT COUNT(*) 
                FROM predictions 
                WHERE timestamp > datetime('now', '-1 day')
            ''')
            recent_predictions = cursor.fetchone()[0]
            
            conn.close()
            
            return {
                'total_predictions': total_predictions,
                'class_distribution': class_counts,
                'average_confidence': round(avg_confidence, 3),
                'recent_predictions_24h': recent_predictions
            }
            
        except Exception as e:
            print(f"Error getting prediction stats: {e}")
            return {
                'total_predictions': 0,
                'class_distribution': {},
                'average_confidence': 0.0,
                'recent_predictions_24h': 0
            }
    
    def save_uploaded_files_for_retraining(self, uploaded_files, save_dir='uploads'):
        """
        Save uploaded files for retraining
        
        Args:
            uploaded_files (list): List of uploaded file objects
            save_dir (str): Directory to save files
            
        Returns:
            list: List of saved file paths
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        saved_paths = []
        
        for i, file in enumerate(uploaded_files):
            try:
                # Generate unique filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"upload_{timestamp}_{i}.jpg"
                file_path = os.path.join(save_dir, filename)
                
                # Save file
                with open(file_path, 'wb') as f:
                    f.write(file.read())
                
                saved_paths.append(file_path)
                print(f"Saved uploaded file: {file_path}")
                
            except Exception as e:
                print(f"Error saving uploaded file: {e}")
        
        return saved_paths 