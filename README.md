# ML Image Classifier Pipeline

A comprehensive Machine Learning pipeline for image classification (Cat vs Dog) with web interface, API endpoints, monitoring, and retraining capabilities.

## 🎯 Project Overview

This project demonstrates a complete end-to-end Machine Learning pipeline including:
- **Data Processing**: Image preprocessing and augmentation
- **Model Training**: Custom CNN and transfer learning models
- **Web Application**: Modern UI for predictions and retraining
- **API Endpoints**: RESTful API for model serving
- **Monitoring**: Real-time system monitoring and metrics
- **Load Testing**: Locust-based performance testing
- **Retraining**: Automated model retraining with new data
- **Local Deployment**: Easy setup and deployment

## 🏗️ Architecture

```
ml-image-classifier-1/
│
├── app.py                 # Main Flask application
├── locustfile.py         # Load testing configuration
├── requirements.txt      # Python dependencies
├── README.md            # Project documentation
│
├── src/                 # Core ML modules
│   ├── preprocessing.py # Data preprocessing and augmentation
│   ├── model.py        # Model architecture and training
│   └── prediction.py   # Prediction service and utilities
│
├── data/               # Dataset
│   ├── train/         # Training images
│   └── test/          # Test images
│
├── models/            # Trained models
│   ├── best_model.h5  # Best trained model
│   └── model_metadata.json # Model metadata
│
├── notebook/          # Jupyter notebooks
│   └── ml-image-classifier.ipynb # Complete ML pipeline
│
├── templates/         # HTML templates
│   └── index.html    # Main dashboard
│
├── static/           # Static assets
│   ├── css/         # Stylesheets
│   └── js/          # JavaScript files
│
└── uploads/          # Uploaded files for retraining
```

## 🚀 Features

### Core ML Pipeline
- **Data Acquisition**: Automated data loading from directory structure
- **Data Processing**: Image preprocessing, augmentation, and normalization
- **Model Creation**: Custom CNN and transfer learning models (VGG16, ResNet50)
- **Model Testing**: Comprehensive evaluation with multiple metrics
- **Model Retraining**: Automated retraining with new data uploads

### Web Application
- **Modern UI**: Responsive Bootstrap-based interface
- **Real-time Monitoring**: System metrics and model performance
- **Prediction Interface**: Upload images and get instant predictions
- **Retraining Interface**: Upload new data and trigger model retraining
- **Visualization**: Training history, confusion matrix, and data analysis

### API Endpoints
- `POST /api/predict` - Make predictions on uploaded images
- `POST /api/upload` - Upload files for retraining
- `POST /api/retrain` - Trigger model retraining
- `GET /api/statistics` - Get system statistics
- `GET /api/predictions` - Get recent predictions
- `GET /api/model-info` - Get model information
- `GET /api/health` - Health check endpoint

### Monitoring & Performance
- **System Monitoring**: CPU, memory, and disk usage
- **Model Metrics**: Accuracy, precision, recall, F1-score
- **Load Testing**: Locust-based performance testing
- **Database Logging**: SQLite database for predictions and training logs

## 📋 Requirements

### System Requirements
- Python 3.8+
- 4GB+ RAM (for model training)
- 2GB+ free disk space

### Python Dependencies
```
tensorflow==2.13.0
flask==2.3.2
opencv-python==4.8.0.76
pillow==10.0.0
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2
locust==2.15.1
psutil==5.9.5
```

## 🛠️ Installation & Setup

### 1. Clone the Repository
   ```bash
   git clone <repository-url>
cd ml-image-classifier-1
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

### 4. Prepare Dataset
Ensure your dataset is organized as follows:
```
data/
├── train/
│   ├── cat.0.jpg
│   ├── cat.1.jpg
│   ├── dog.0.jpg
│   └── dog.1.jpg
└── test/
    ├── cat.10.jpg
    ├── cat.11.jpg
    ├── dog.10.jpg
    └── dog.11.jpg
```

### 5. Train Initial Model
```bash
# Run the Jupyter notebook to train the model
jupyter notebook notebook/ml-image-classifier.ipynb
```

## 🚀 Usage

### Starting the Application
   ```bash
   python app.py
```

The application will be available at:
- **Web Interface**: http://localhost:5000
- **API Base URL**: http://localhost:5000/api

### Using the Web Interface

1. **Prediction Tab**
   - Upload an image file
   - Click "Predict" to get classification results
   - View confidence scores and class probabilities

2. **Retrain Tab**
   - Upload multiple images for retraining
   - Click "Start Retraining" to train a new model
   - Monitor training progress in real-time

3. **Monitoring Tab**
   - View system statistics (CPU, memory, disk usage)
   - Check recent predictions and model performance
   - Monitor model information and status

### Using the API

#### Make a Prediction
```bash
curl -X POST -F "file=@image.jpg" http://localhost:5000/api/predict
```

Response:
```json
{
  "predicted_class": "cat",
  "confidence": 0.95,
  "probabilities": {
    "cat": 0.95,
    "dog": 0.05
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

#### Upload Files for Retraining
   ```bash
curl -X POST -F "files=@image1.jpg" -F "files=@image2.jpg" http://localhost:5000/api/upload
```

#### Trigger Model Retraining
   ```bash
curl -X POST http://localhost:5000/api/retrain
   ```

#### Get System Statistics
   ```bash
curl http://localhost:5000/api/statistics
   ```

## 📊 Load Testing

### Running Load Tests with Locust

1. **Start the application** (if not already running):
   ```bash
   python app.py
   ```

2. **Run Locust**:
   ```bash
   locust -f locustfile.py --host=http://localhost:5000
   ```

3. **Open Locust Web Interface**:
   - Go to http://localhost:8089
   - Configure number of users and spawn rate
   - Start the test

### Predefined Test Scenarios

```bash
# Light load test (10 users)
locust -f locustfile.py --host=http://localhost:5000 --users=10 --spawn-rate=1 --run-time=2m

# Medium load test (50 users)
locust -f locustfile.py --host=http://localhost:5000 --users=50 --spawn-rate=5 --run-time=5m

# Heavy load test (100 users)
locust -f locustfile.py --host=http://localhost:5000 --users=100 --spawn-rate=10 --run-time=10m

# Stress test (200 users)
locust -f locustfile.py --host=http://localhost:5000 --users=200 --spawn-rate=20 --run-time=15m
```

## 📈 Model Performance

### Evaluation Metrics
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall

### Expected Performance
- **Custom CNN**: ~85-90% accuracy
- **VGG16 Transfer Learning**: ~90-95% accuracy
- **ResNet50 Transfer Learning**: ~92-97% accuracy

## 🔧 Configuration

### Environment Variables
```bash
export FLASK_ENV=development
export FLASK_DEBUG=1
export MODEL_PATH=models/best_model.h5
export UPLOAD_FOLDER=uploads
```

### Model Configuration
Edit `src/model.py` to modify:
- Image size (default: 224x224)
- Model architecture
- Training parameters
- Data augmentation settings

## 🚀 Deployment

### Local Development
   ```bash
# Install dependencies
   pip install -r requirements.txt

# Run the application
python app.py

# Access the web interface
# Open http://localhost:5000 in your browser
```

### Production Deployment
For production deployment, consider using:
- **Gunicorn** with WSGI server
- **Nginx** as reverse proxy
- **Cloud platforms** like Heroku, AWS, or Google Cloud

## 📝 API Documentation

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web interface |
| POST | `/api/predict` | Make prediction |
| POST | `/api/upload` | Upload files |
| POST | `/api/retrain` | Trigger retraining |
| GET | `/api/statistics` | System statistics |
| GET | `/api/predictions` | Recent predictions |
| GET | `/api/model-info` | Model information |
| GET | `/api/health` | Health check |

### Request/Response Examples

See the [API Examples](docs/api-examples.md) for detailed request/response formats.

## 🔍 Troubleshooting

### Common Issues

1. **Model not loading**
   - Ensure model file exists in `models/` directory
   - Check model metadata file
   - Verify TensorFlow version compatibility

2. **Memory errors during training**
   - Reduce batch size in `src/preprocessing.py`
   - Use smaller image size
   - Close other applications

3. **Upload errors**
   - Check file size limits (16MB max)
   - Verify file format (jpg, png, bmp, tiff)
   - Ensure upload directory has write permissions

4. **API connection errors**
   - Verify Flask app is running
   - Check port 5000 is available
   - Ensure firewall allows connections

### Debug Mode
```bash
export FLASK_DEBUG=1
python app.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow and Keras for deep learning framework
- Flask for web framework
- Bootstrap for UI components
- Locust for load testing
- OpenCV for image processing

## 📞 Support

For support and questions:
- Create an issue in the repository
- Contact: [your-email@example.com]
- Documentation: [link-to-docs]

---

**Note**: This project is designed for educational and demonstration purposes. For production use, consider additional security measures, error handling, and scalability improvements. 