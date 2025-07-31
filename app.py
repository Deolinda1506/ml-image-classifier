import os
import json
import threading
import time
from datetime import datetime
from flask import Flask, request, jsonify, render_template, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import numpy as np
import psutil
import sqlite3

# Import our custom modules
from src.preprocessing import ImagePreprocessor
from src.model import ImageClassifier
from src.prediction import PredictionService

app = Flask(__name__)
CORS(app)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['MODEL_PATH'] = 'models/best_model.h5'
app.config['METADATA_PATH'] = 'models/model_metadata.json'

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('logs', exist_ok=True)

# Global variables
prediction_service = None
training_in_progress = False
training_status = {
    'status': 'idle',
    'progress': 0,
    'message': '',
    'start_time': None,
    'end_time': None,
    'metrics': {}
}

# Initialize database
def init_database():
    """Initialize SQLite database for storing predictions and training logs"""
    conn = sqlite3.connect('ml_pipeline.db')
    cursor = conn.cursor()
    
    # Create predictions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            predicted_class TEXT,
            confidence REAL,
            image_path TEXT,
            error TEXT,
            model_path TEXT
        )
    ''')
    
    # Create training_logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            status TEXT,
            message TEXT,
            metrics TEXT,
            duration REAL
        )
    ''')
    
    # Create uploaded_files table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS uploaded_files (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            original_name TEXT,
            file_path TEXT,
            file_size INTEGER,
            uploaded_at TEXT,
            processed BOOLEAN DEFAULT FALSE
        )
    ''')
    
    conn.commit()
    conn.close()

# Initialize database on startup
init_database()

def load_prediction_service():
    """Load the prediction service"""
    global prediction_service
    try:
        if os.path.exists(app.config['MODEL_PATH']):
            prediction_service = PredictionService(
                app.config['MODEL_PATH'],
                app.config['METADATA_PATH']
            )
            print("Prediction service loaded successfully!")
        else:
            print("No trained model found. Please train a model first.")
    except Exception as e:
        print(f"Error loading prediction service: {e}")

# Load prediction service on startup
load_prediction_service()

def save_prediction_to_db(prediction):
    """Save prediction to database"""
    conn = sqlite3.connect('ml_pipeline.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO predictions (timestamp, predicted_class, confidence, image_path, error, model_path)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (
        prediction.get('timestamp'),
        prediction.get('predicted_class'),
        prediction.get('confidence'),
        prediction.get('image_path'),
        prediction.get('error'),
        prediction.get('model_path')
    ))
    
    conn.commit()
    conn.close()

def save_training_log(status, message, metrics=None, duration=None):
    """Save training log to database"""
    conn = sqlite3.connect('ml_pipeline.db')
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO training_logs (timestamp, status, message, metrics, duration)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        datetime.now().isoformat(),
        status,
        message,
        json.dumps(metrics) if metrics else None,
        duration
    ))
    
    conn.commit()
    conn.close()

def train_model_async():
    """Train model in background thread"""
    global training_in_progress, training_status, prediction_service
    
    try:
        training_in_progress = True
        training_status.update({
            'status': 'training',
            'progress': 0,
            'message': 'Loading data...',
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'metrics': {}
        })
        
        save_training_log('started', 'Training started')
        
        # Load and preprocess data
        training_status['message'] = 'Loading and preprocessing data...'
        training_status['progress'] = 10
        
        preprocessor = ImagePreprocessor()
        X_train, y_train, X_test, y_test = preprocessor.load_dataset_from_flat_structure('data')
        
        # Split training data into train and validation
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        training_status['message'] = 'Creating data generators...'
        training_status['progress'] = 20
        
        # Create data generators
        train_generator, val_generator = preprocessor.create_data_generators(X_train, y_train, X_val, y_val)
        
        # Build and train model
        training_status['message'] = 'Building model...'
        training_status['progress'] = 30
        
        classifier = ImageClassifier(model_type='custom')
        classifier.build_model()
        
        training_status['message'] = 'Training model...'
        training_status['progress'] = 40
        
        # Train the model
        history = classifier.train(
            train_generator, 
            val_generator, 
            epochs=50,
            model_save_path=app.config['MODEL_PATH']
        )
        
        training_status['message'] = 'Evaluating model...'
        training_status['progress'] = 80
        
        # Evaluate model
        metrics = classifier.evaluate_model(X_test, y_test)
        
        # Save model metadata
        classifier.save_model(
            app.config['MODEL_PATH'],
            app.config['METADATA_PATH']
        )
        
        # Reload prediction service
        load_prediction_service()
        
        training_status.update({
            'status': 'completed',
            'progress': 100,
            'message': 'Training completed successfully!',
            'end_time': datetime.now().isoformat(),
            'metrics': metrics
        })
        
        duration = (datetime.fromisoformat(training_status['end_time']) - 
                   datetime.fromisoformat(training_status['start_time'])).total_seconds()
        
        save_training_log('completed', 'Training completed successfully', metrics, duration)
        
    except Exception as e:
        training_status.update({
            'status': 'failed',
            'progress': 0,
            'message': f'Training failed: {str(e)}',
            'end_time': datetime.now().isoformat(),
            'metrics': {}
        })
        
        save_training_log('failed', f'Training failed: {str(e)}')
        
    finally:
        training_in_progress = False

# API Routes

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction on uploaded image"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if prediction_service is None:
            return jsonify({'error': 'Model not loaded. Please train a model first.'}), 500
        
        # Make prediction
        prediction = prediction_service.predict_from_file(file)
        
        # Save to database
        save_prediction_to_db(prediction)
        
        return jsonify(prediction)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Upload files for retraining"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('files')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        if prediction_service is None:
            return jsonify({'error': 'Prediction service not initialized'}), 500
        
        # Process uploaded files
        result = prediction_service.process_bulk_upload(files, app.config['UPLOAD_FOLDER'])
        
        # Save to database
        conn = sqlite3.connect('ml_pipeline.db')
        cursor = conn.cursor()
        
        for file_info in result['processed_files']:
            cursor.execute('''
                INSERT INTO uploaded_files (filename, original_name, file_path, file_size, uploaded_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                file_info['saved_path'].split('/')[-1],
                file_info['original_name'],
                file_info['saved_path'],
                file_info['file_size'],
                file_info['uploaded_at']
            ))
        
        conn.commit()
        conn.close()
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/retrain', methods=['POST'])
def retrain():
    """Trigger model retraining"""
    global training_in_progress
    
    if training_in_progress:
        return jsonify({'error': 'Training already in progress'}), 400
    
    # Start training in background thread
    thread = threading.Thread(target=train_model_async)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'message': 'Training started',
        'status': 'training'
    })

@app.route('/api/training-status')
def training_status_api():
    """Get current training status"""
    return jsonify(training_status)

@app.route('/api/model-info')
def model_info():
    """Get model information"""
    if prediction_service is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify(prediction_service.get_model_info())

@app.route('/api/predictions')
def get_predictions():
    """Get recent predictions"""
    try:
        conn = sqlite3.connect('ml_pipeline.db')
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT 100
        ''')
        
        rows = cursor.fetchall()
        columns = [description[0] for description in cursor.description]
        
        predictions = []
        for row in rows:
            predictions.append(dict(zip(columns, row)))
        
        conn.close()
        
        return jsonify(predictions)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/statistics')
def get_statistics():
    """Get system statistics"""
    try:
        # System stats
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Prediction stats
        conn = sqlite3.connect('ml_pipeline.db')
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM predictions WHERE error IS NULL')
        successful_predictions = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM uploaded_files')
        total_uploads = cursor.fetchone()[0]
        
        conn.close()
        
        stats = {
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': (disk.used / disk.total) * 100
            },
            'predictions': {
                'total': total_predictions,
                'successful': successful_predictions,
                'success_rate': (successful_predictions / total_predictions * 100) if total_predictions > 0 else 0
            },
            'uploads': {
                'total': total_uploads
            },
            'training': training_status
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': prediction_service is not None,
        'training_in_progress': training_in_progress
    })

# Static file serving
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 