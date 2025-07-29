#!/usr/bin/env python3
"""
🐱🐶 ML Image Classification API

Flask REST API for image classification of cats and dogs.
Provides endpoints for predictions, model training, and monitoring.
"""

import os
import json
import time
import threading
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tensorflow as tf

# Import our custom modules
from src.model import ImageClassifier
from src.prediction import ImagePredictor
from src.preprocessing import ImagePreprocessor

# Initialize Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'models/image_classifier.h5'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Create directories if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('models', exist_ok=True)

# Global variables
predictor = None
classifier = None
preprocessor = None
training_thread = None
training_status = {
    'is_training': False,
    'progress': 0,
    'current_epoch': 0,
    'total_epochs': 0,
    'accuracy': 0,
    'loss': 0,
    'start_time': None,
    'end_time': None
}

# Metrics tracking
prediction_history = []
system_metrics = {
    'start_time': datetime.now(),
    'total_predictions': 0,
    'successful_predictions': 0,
    'failed_predictions': 0,
    'average_latency': 0
}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def initialize_model():
    """Initialize the model and predictor"""
    global predictor, classifier, preprocessor
    
    try:
        # Initialize predictor if model exists
        if os.path.exists(MODEL_PATH):
            predictor = ImagePredictor(MODEL_PATH)
            print("✅ Model loaded successfully")
        else:
            print("⚠️ No trained model found. Train the model first.")
        
        # Initialize classifier and preprocessor
        classifier = ImageClassifier()
        preprocessor = ImagePreprocessor()
        print("✅ Components initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing model: {e}")

def update_training_status(progress, epoch, total_epochs, accuracy, loss):
    """Update training status"""
    global training_status
    training_status.update({
        'progress': progress,
        'current_epoch': epoch,
        'total_epochs': total_epochs,
        'accuracy': accuracy,
        'loss': loss
    })

def train_model_async():
    """Train model in background thread"""
    global training_status, predictor
    
    try:
        training_status['is_training'] = True
        training_status['start_time'] = datetime.now()
        
        # Prepare data
        train_generator, validation_generator = preprocessor.prepare_data()
        
        # Build and train model
        model = classifier.build_model()
        
        # Custom callback to update status
        class StatusCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                progress = ((epoch + 1) / training_status['total_epochs']) * 100
                update_training_status(
                    progress=progress,
                    epoch=epoch + 1,
                    total_epochs=training_status['total_epochs'],
                    accuracy=logs.get('accuracy', 0),
                    loss=logs.get('loss', 0)
                )
        
        # Train model
        history = classifier.train(
            train_generator, 
            validation_generator, 
            epochs=training_status['total_epochs'],
            callbacks=[StatusCallback()]
        )
        
        # Save model
        classifier.save_model(MODEL_PATH)
        
        # Reload predictor with new model
        predictor = ImagePredictor(MODEL_PATH)
        
        training_status['end_time'] = datetime.now()
        training_status['is_training'] = False
        
        print("✅ Model training completed successfully")
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        training_status['is_training'] = False

# Initialize model on startup
initialize_model()

@app.route('/')
def index():
    """API documentation"""
    return jsonify({
        'name': 'ML Image Classification API',
        'version': '1.0.0',
        'description': 'Flask REST API for cat/dog image classification',
        'endpoints': {
            'GET /': 'API documentation',
            'POST /predict': 'Make predictions on images',
            'GET /status': 'Get model status and metrics',
            'POST /train': 'Start model training',
            'GET /training-status': 'Get training progress',
            'POST /upload-data': 'Upload new data for retraining',
            'POST /retrain': 'Retrain model with new data',
            'GET /health': 'Health check'
        }
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Make prediction on uploaded image"""
    global predictor, system_metrics, prediction_history
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Make prediction
        start_time = time.time()
        
        if predictor is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
        
        result = predictor.predict(filepath)
        prediction_time = time.time() - start_time
        
        # Update metrics
        system_metrics['total_predictions'] += 1
        system_metrics['successful_predictions'] += 1
        
        # Update average latency
        total_latency = system_metrics['average_latency'] * (system_metrics['successful_predictions'] - 1)
        system_metrics['average_latency'] = (total_latency + prediction_time) / system_metrics['successful_predictions']
        
        # Add to prediction history
        prediction_history.append({
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'predicted_class': result['predicted_class'],
            'confidence': result['confidence'],
            'prediction_time': prediction_time
        })
        
        # Keep only last 100 predictions
        if len(prediction_history) > 100:
            prediction_history.pop(0)
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return jsonify({
            'predicted_class': result['predicted_class'],
            'confidence': result['confidence'],
            'prediction_time': prediction_time,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        system_metrics['failed_predictions'] += 1
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/status')
def status():
    """Get model status and metrics"""
    global predictor, system_metrics, training_status
    
    model_info = {}
    if predictor is not None:
        try:
            model_info = predictor.get_model_info()
        except:
            model_info = {'error': 'Could not get model info'}
    
    uptime = datetime.now() - system_metrics['start_time']
    
    return jsonify({
        'model_loaded': predictor is not None,
        'model_info': model_info,
        'system_metrics': {
            'uptime_seconds': uptime.total_seconds(),
            'total_predictions': system_metrics['total_predictions'],
            'successful_predictions': system_metrics['successful_predictions'],
            'failed_predictions': system_metrics['failed_predictions'],
            'success_rate': (system_metrics['successful_predictions'] / max(system_metrics['total_predictions'], 1)) * 100,
            'average_latency_ms': system_metrics['average_latency'] * 1000
        },
        'training_status': training_status,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/train', methods=['POST'])
def train():
    """Start model training"""
    global training_thread, training_status
    
    if training_status['is_training']:
        return jsonify({'error': 'Training already in progress'}), 400
    
    try:
        # Get training parameters from request
        data = request.get_json() or {}
        epochs = data.get('epochs', 50)
        
        training_status['total_epochs'] = epochs
        training_status['progress'] = 0
        training_status['current_epoch'] = 0
        
        # Start training in background thread
        training_thread = threading.Thread(target=train_model_async)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            'message': 'Training started successfully',
            'epochs': epochs,
            'status': 'started'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to start training: {str(e)}'}), 500

@app.route('/training-status')
def get_training_status():
    """Get current training status"""
    global training_status
    
    return jsonify(training_status)

@app.route('/upload-data', methods=['POST'])
def upload_data():
    """Upload new data for retraining"""
    if 'files[]' not in request.files:
        return jsonify({'error': 'No files provided'}), 400
    
    files = request.files.getlist('files[]')
    uploaded_files = []
    
    for file in files:
        if file.filename == '':
            continue
        
        if not allowed_file(file.filename):
            continue
        
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            uploaded_files.append(filename)
        except Exception as e:
            return jsonify({'error': f'Failed to save file {file.filename}: {str(e)}'}), 500
    
    return jsonify({
        'message': f'Successfully uploaded {len(uploaded_files)} files',
        'uploaded_files': uploaded_files
    })

@app.route('/retrain', methods=['POST'])
def retrain():
    """Retrain model with new data"""
    global training_thread, training_status
    
    if training_status['is_training']:
        return jsonify({'error': 'Training already in progress'}), 400
    
    try:
        data = request.get_json()
        if not data or 'files' not in data:
            return jsonify({'error': 'No files specified for retraining'}), 400
        
        # Start retraining process
        epochs = data.get('epochs', 10)
        training_status['total_epochs'] = epochs
        training_status['progress'] = 0
        training_status['current_epoch'] = 0
        
        # Start training in background thread
        training_thread = threading.Thread(target=train_model_async)
        training_thread.daemon = True
        training_thread.start()
        
        return jsonify({
            'message': 'Retraining started successfully',
            'epochs': epochs,
            'status': 'started'
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to start retraining: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': predictor is not None
    })

@app.route('/predictions')
def get_predictions():
    """Get recent prediction history"""
    global prediction_history
    
    return jsonify({
        'predictions': prediction_history[-50:],  # Last 50 predictions
        'total_count': len(prediction_history)
    })

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Image Classification API')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the server on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run the server on')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    print("🚀 Starting ML Image Classification API...")
    print(f"📡 Server will be available at http://{args.host}:{args.port}")
    print("📊 Dashboard will be available at http://localhost:8050")
    
    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
        threaded=True
    ) 