import os
import sys
import sqlite3
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import psutil
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append('src')

# Import TensorFlow and our custom modules
import tensorflow as tf
from tensorflow import keras
from preprocessing import ImagePreprocessor
from model import ImageClassifier
from prediction import PredictionService

from flask import Flask, request, jsonify, render_template, send_file
from flask_cors import CORS
import base64
from io import BytesIO
import traceback

app = Flask(__name__)
CORS(app)

# Global variables
MODEL_PATH = 'models/best_tensorflow_cnn_model.h5'
classifier = None
preprocessor = None
prediction_service = None

def init_database():
    """Initialize SQLite database for predictions and training logs"""
    conn = sqlite3.connect('predictions.db')
    cursor = conn.cursor()
    
    # Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            image_name TEXT,
            predicted_class TEXT,
            confidence REAL,
            actual_class TEXT,
            processing_time REAL
        )
    ''')
    
    # Create training_logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            model_type TEXT,
            accuracy REAL,
            precision REAL,
            recall REAL,
            f1_score REAL,
            training_time REAL,
            dataset_size INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()

def load_model():
    """Load the trained TensorFlow CNN model"""
    global classifier, preprocessor, prediction_service
    
    try:
        print("Loading TensorFlow CNN model...")
        
        # Initialize preprocessor
        preprocessor = ImagePreprocessor(img_size=(128, 128), batch_size=32)
        
        # Initialize classifier
        classifier = ImageClassifier(model_type='cnn', img_size=(128, 128), num_classes=2)
        
        # Load model if it exists
        if os.path.exists(MODEL_PATH):
            classifier.load_model(MODEL_PATH)
            print(f"Model loaded from {MODEL_PATH}")
        else:
            print("No pre-trained model found. You need to train the model first.")
            return False
        
        # Initialize prediction service
        prediction_service = PredictionService(MODEL_PATH)
        
        print("TensorFlow CNN model loaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Record start time
        start_time = datetime.now()
        
        # Make prediction
        prediction_result = prediction_service.predict_uploaded_image(file)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Log prediction to database
        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO predictions (image_name, predicted_class, confidence, processing_time)
            VALUES (?, ?, ?, ?)
        ''', (file.filename, prediction_result['prediction'], prediction_result['confidence'], processing_time))
        conn.commit()
        conn.close()
        
        # Add processing time to response
        prediction_result['processing_time'] = processing_time
        
        return jsonify(prediction_result)
        
    except Exception as e:
        print(f"Error in prediction: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrain', methods=['POST'])
def retrain():
    """API endpoint for retraining the model"""
    try:
        if 'data' not in request.files:
            return jsonify({'error': 'No data file provided'}), 400
        
        data_file = request.files['data']
        
        # Record start time
        start_time = datetime.now()
        
        # Load and preprocess new data
        print("Loading new training data...")
        X_train, y_train, X_test, y_test = preprocessor.load_dataset_from_flat_structure('data')
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        # Retrain model
        print("Retraining TensorFlow CNN model...")
        history = classifier.train(
            X_train_split, y_train_split, X_val, y_val,
            epochs=30, batch_size=32, model_save_path=MODEL_PATH
        )
        
        # Evaluate model
        metrics = classifier.evaluate(X_test, y_test)
        
        # Calculate training time
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Log training to database
        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO training_logs (model_type, accuracy, precision, recall, f1_score, training_time, dataset_size)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', ('tensorflow_cnn', metrics['accuracy'], metrics['precision'], 
              metrics['recall'], metrics['f1_score'], training_time, len(X_train)))
        conn.commit()
        conn.close()
        
        return jsonify({
            'message': 'Model retrained successfully',
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'training_time': training_time,
            'dataset_size': len(X_train)
        })
        
    except Exception as e:
        print(f"Error in retraining: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def status():
    """API endpoint for system status"""
    try:
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Get model info
        model_info = classifier.get_model_info() if classifier else None
        
        # Get recent predictions
        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM predictions WHERE timestamp >= datetime("now", "-1 hour")')
        recent_predictions = cursor.fetchone()[0]
        
        cursor.execute('SELECT AVG(processing_time) FROM predictions')
        avg_processing_time = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return jsonify({
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': (disk.used / disk.total) * 100
            },
            'model': model_info,
            'predictions': {
                'total': total_predictions,
                'recent_1h': recent_predictions,
                'avg_processing_time': avg_processing_time
            },
            'status': 'healthy'
        })
        
    except Exception as e:
        print(f"Error in status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/predictions')
def get_predictions():
    """API endpoint to get recent predictions"""
    try:
        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, image_name, predicted_class, confidence, processing_time
            FROM predictions
            ORDER BY timestamp DESC
            LIMIT 50
        ''')
        predictions = cursor.fetchall()
        conn.close()
        
        return jsonify([{
            'timestamp': pred[0],
            'image_name': pred[1],
            'predicted_class': pred[2],
            'confidence': pred[3],
            'processing_time': pred[4]
        } for pred in predictions])
        
    except Exception as e:
        print(f"Error getting predictions: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/training_history')
def get_training_history():
    """API endpoint to get training history"""
    try:
        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()
        cursor.execute('''
            SELECT timestamp, model_type, accuracy, precision, recall, f1_score, training_time, dataset_size
            FROM training_logs
            ORDER BY timestamp DESC
            LIMIT 20
        ''')
        training_logs = cursor.fetchall()
        conn.close()
        
        return jsonify([{
            'timestamp': log[0],
            'model_type': log[1],
            'accuracy': log[2],
            'precision': log[3],
            'recall': log[4],
            'f1_score': log[5],
            'training_time': log[6],
            'dataset_size': log[7]
        } for log in training_logs])
        
    except Exception as e:
        print(f"Error getting training history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/visualizations')
def get_visualizations():
    """API endpoint to generate and return visualizations"""
    try:
        # Get prediction data for visualization
        conn = sqlite3.connect('predictions.db')
        cursor = conn.cursor()
        cursor.execute('SELECT predicted_class, confidence, timestamp FROM predictions')
        data = cursor.fetchall()
        conn.close()
        
        if not data:
            return jsonify({'error': 'No prediction data available'}), 404
        
        # Create visualizations
        df = pd.DataFrame(data, columns=['predicted_class', 'confidence', 'timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # 1. Class distribution
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        df['predicted_class'].value_counts().plot(kind='bar')
        plt.title('Prediction Class Distribution')
        plt.ylabel('Count')
        
        # 2. Confidence distribution
        plt.subplot(2, 3, 2)
        plt.hist(df['confidence'], bins=20, alpha=0.7)
        plt.title('Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        
        # 3. Processing time distribution
        plt.subplot(2, 3, 3)
        cursor.execute('SELECT processing_time FROM predictions')
        processing_times = [row[0] for row in cursor.fetchall()]
        plt.hist(processing_times, bins=20, alpha=0.7)
        plt.title('Processing Time Distribution')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Frequency')
        
        # 4. Predictions over time
        plt.subplot(2, 3, 4)
        df.set_index('timestamp')['predicted_class'].value_counts().plot(kind='line')
        plt.title('Predictions Over Time')
        plt.ylabel('Count')
        
        # 5. Confidence by class
        plt.subplot(2, 3, 5)
        df.boxplot(column='confidence', by='predicted_class')
        plt.title('Confidence by Class')
        plt.suptitle('')
        
        # 6. System metrics
        plt.subplot(2, 3, 6)
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        metrics = ['CPU', 'Memory', 'Disk']
        values = [cpu_percent, memory.percent, psutil.disk_usage('/').percent]
        plt.bar(metrics, values)
        plt.title('System Metrics')
        plt.ylabel('Percentage')
        
        plt.tight_layout()
        
        # Save plot to bytes
        img_buffer = BytesIO()
        plt.savefig(img_buffer, format='png', dpi=300, bbox_inches='tight')
        img_buffer.seek(0)
        plt.close()
        
        # Encode to base64
        img_str = base64.b64encode(img_buffer.getvalue()).decode()
        
        return jsonify({
            'visualization': f'data:image/png;base64,{img_str}'
        })
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize database
    init_database()
    
    # Load model
    if load_model():
        print("Starting Flask application...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load model. Please train the model first.")
        print("Run: python notebook/ml_pipeline_tensorflow.py") 