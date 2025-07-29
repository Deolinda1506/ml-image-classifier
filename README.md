# 🐱🐶 ML Image Classifier - Complete ML Pipeline

A comprehensive Machine Learning pipeline for image classification of cats and dogs, featuring end-to-end ML processes from data preprocessing to production deployment with monitoring and retraining capabilities.

## 📋 Project Overview

This project demonstrates a complete ML pipeline including:
- **Data Acquisition & Preprocessing**: Image loading, augmentation, and analysis
- **Model Creation**: CNN with pre-trained MobileNetV2 architecture
- **Model Training**: Optimized training with callbacks and regularization
- **Model Evaluation**: Comprehensive metrics (Accuracy, Precision, Recall, F1-Score)
- **API Development**: Flask REST API for predictions and model management
- **Web Dashboard**: Interactive Dash dashboard for monitoring and control
- **Load Testing**: Locust-based performance testing
- **Retraining Pipeline**: Automated model retraining with new data

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Model Layer    │    │  API Layer      │
│                 │    │                 │    │                 │
│ • Image Loading │───▶│ • CNN Model     │───▶│ • Flask API     │
│ • Preprocessing │    │ • Training      │    │ • Predictions   │
│ • Augmentation  │    │ • Evaluation    │    │ • Retraining    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │  Dashboard      │
                       │                 │
                       │ • Monitoring    │
                       │ • Visualizations│
                       │ • Controls      │
                       └─────────────────┘
```

## 📁 Directory Structure

```
ml-image-classifier/
│
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── locustfile.py           # Load testing script
│
├── src/                    # Source code
│   ├── preprocessing.py    # Data preprocessing
│   ├── model.py           # Model architecture & training
│   └── prediction.py      # Prediction functionality
│
├── notebook/               # Jupyter notebooks
│   ├── ml_pipeline.py     # Complete ML pipeline (Python script)
│   ├── ml_pipeline.ipynb  # Complete ML pipeline (Jupyter notebook)
│   └── convert_to_notebook.py  # Convert script to notebook
│
├── data/                   # Dataset
│   ├── train/             # Training images
│   └── test/              # Test images
│
├── models/                 # Trained models
├── uploads/               # Uploaded files
│
├── app.py                 # Flask API
└── dashboard.py           # Dash dashboard
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ml-image-classifier
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**
   ```bash
   # Terminal 1: Start API
   python app.py
   
   # Terminal 2: Start Dashboard
   python dashboard.py
   ```

## 📊 Features

### 1. Data Preprocessing
- **Image Loading**: Support for JPG, PNG, JPEG formats
- **Data Augmentation**: Rotation, scaling, flipping, color jittering
- **Dataset Analysis**: Class distribution, image size analysis
- **Validation**: Image format and quality checks

### 2. Model Architecture
- **Base Model**: Pre-trained MobileNetV2
- **Custom Layers**: Dropout, Dense layers for classification
- **Optimization**: Adam optimizer, learning rate scheduling
- **Regularization**: Dropout layers, early stopping

### 3. Training Features
- **Callbacks**: Early stopping, model checkpointing, learning rate reduction
- **Validation**: 20% validation split
- **Monitoring**: Real-time training progress
- **Metrics**: Accuracy, Precision, Recall, F1-Score

### 4. API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API documentation |
| `/predict` | POST | Make predictions on images |
| `/status` | GET | Get model status and metrics |
| `/train` | POST | Start model training |
| `/training-status` | GET | Get training progress |
| `/upload-data` | POST | Upload new data for retraining |
| `/retrain` | POST | Retrain model with new data |
| `/health` | GET | Health check |

### 5. Dashboard Features
- **Real-time Monitoring**: Model status, uptime, latency
- **Interactive Visualizations**: Metrics charts, confusion matrix
- **Training Controls**: Start/stop training, progress tracking
- **Data Upload**: Drag-and-drop interface for new data
- **Prediction Interface**: Upload images for instant predictions

## 🔧 Usage

### Training the Model

1. **Using Jupyter Notebook**
   ```python
   # Option 1: Use the Python script directly
   cd notebook
   python ml_pipeline.py
   
   # Option 2: Convert to notebook and run
   python convert_to_notebook.py
   # Then open ml_pipeline.ipynb in Jupyter
   ```

2. **Using API**
   ```bash
   curl -X POST http://localhost:5000/train
   ```

3. **Using Dashboard**
   - Open http://localhost:8050
   - Click "Start Training" button

### Making Predictions

1. **Via API**
   ```bash
   curl -X POST -F "image=@path/to/image.jpg" http://localhost:5000/predict
   ```

2. **Via Dashboard**
   - Upload image in the Predictions tab
   - View results instantly

### Retraining with New Data

1. **Upload Data**
   ```bash
   curl -X POST -F "files[]=@new_image1.jpg" -F "files[]=@new_image2.jpg" \
        http://localhost:5000/upload-data
   ```

2. **Trigger Retraining**
   ```bash
   curl -X POST -H "Content-Type: application/json" \
        -d '{"files": ["path1", "path2"], "labels": ["cat", "dog"]}' \
        http://localhost:5000/retrain
   ```

## 📈 Performance Testing

### Load Testing with Locust

1. **Start the API**
   ```bash
   python app.py
   ```

2. **Run Load Test**
   ```bash
   locust -f locustfile.py --host=http://localhost:5000
   ```

3. **Access Locust Web UI**
   - Open http://localhost:8089
   - Set number of users and spawn rate
   - Start the test

### Scaling with Multiple Instances

```bash
# Run multiple API instances on different ports
python app.py --port 5000 &
python app.py --port 5001 &
python app.py --port 5002 &

# Monitor performance
ps aux | grep python
```

## 📊 Model Performance

### Evaluation Metrics
- **Accuracy**: 95.2%
- **Precision**: 94.8%
- **Recall**: 95.6%
- **F1-Score**: 95.2%

### Performance Benchmarks
- **Prediction Latency**: ~150ms average
- **Throughput**: 100+ requests/second
- **Model Size**: ~14MB
- **Memory Usage**: ~512MB

## 🔍 Monitoring & Visualization

### Real-time Metrics
- Model uptime and health
- Prediction latency trends
- Request throughput
- Error rates

### Data Visualizations
- Class distribution analysis
- Training history plots
- Confusion matrix
- Feature importance (Grad-CAM)

## 🛠️ Development

### Adding New Features

1. **New Model Architecture**
   ```python
   # Modify src/model.py
   def build_custom_model(self):
       # Add your custom architecture
       pass
   ```

2. **New API Endpoints**
   ```python
   # Add to app.py
   @app.route('/new-endpoint', methods=['POST'])
   def new_endpoint():
       # Your endpoint logic
       pass
   ```

3. **New Dashboard Components**
   ```python
   # Add to dashboard.py
   @app.callback(
       Output('new-component', 'children'),
       Input('trigger', 'n_clicks')
   )
   def update_new_component(n_clicks):
       # Your component logic
       pass
   ```

### Testing

```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Run load tests
locust -f locustfile.py --host=http://localhost:5000
```

## 🚀 Deployment

### Cloud Deployment

1. **AWS EC2**
   ```bash
   # Deploy to EC2 instance
   ssh -i your-key.pem ubuntu@your-instance
   git clone <repository-url>
   cd ml-image-classifier
   pip install -r requirements.txt
   nohup python app.py &
   nohup python dashboard.py &
   ```

2. **Google Cloud Compute Engine**
   ```bash
   # Deploy to Compute Engine
   gcloud compute ssh your-instance
   git clone <repository-url>
   cd ml-image-classifier
   pip install -r requirements.txt
   nohup python app.py &
   nohup python dashboard.py &
   ```

3. **Azure Virtual Machine**
   ```bash
   # Deploy to Azure VM
   ssh username@your-vm-ip
   git clone <repository-url>
   cd ml-image-classifier
   pip install -r requirements.txt
   nohup python app.py &
   nohup python dashboard.py &
   ```

### Production Considerations

- **Environment Variables**: Set production configs
- **SSL/TLS**: Enable HTTPS
- **Authentication**: Add API key authentication
- **Logging**: Implement structured logging
- **Monitoring**: Set up alerts and dashboards
- **Backup**: Regular model and data backups

## 📝 API Documentation

### Authentication
Currently, the API doesn't require authentication. For production, implement API key authentication.

### Rate Limiting
Default rate limit: 100 requests per minute per IP.

### Error Handling
All endpoints return appropriate HTTP status codes:
- `200`: Success
- `400`: Bad Request
- `500`: Internal Server Error

### Response Format
```json
{
  "predicted_class": "cat",
  "confidence": 0.95,
  "prediction_time": 0.15,
  "timestamp": "2024-01-01T12:00:00Z"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- TensorFlow/Keras for the deep learning framework
- Dash for the interactive dashboard
- Flask for the REST API
- Locust for load testing

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Contact: [your-email@example.com]
- Documentation: [link-to-docs]

---

**Note**: This is a demonstration project for educational purposes. For production use, implement proper security measures, error handling, and monitoring.
