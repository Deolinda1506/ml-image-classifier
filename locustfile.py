from locust import HttpUser, task, between
import random
import os

class MLPipelineUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize user session"""
        # Test if the service is running
        response = self.client.get("/api/health")
        if response.status_code != 200:
            print("Warning: Service might not be running properly")
    
    @task(3)
    def health_check(self):
        """Health check endpoint - high frequency"""
        self.client.get("/api/health")
    
    @task(2)
    def get_statistics(self):
        """Get system statistics"""
        self.client.get("/api/statistics")
    
    @task(2)
    def get_model_info(self):
        """Get model information"""
        self.client.get("/api/model-info")
    
    @task(1)
    def get_predictions(self):
        """Get recent predictions"""
        self.client.get("/api/predictions")
    
    @task(1)
    def get_training_status(self):
        """Get training status"""
        self.client.get("/api/training-status")
    
    @task(1)
    def upload_files(self):
        """Upload files for retraining"""
        # Create a dummy image file for testing
        dummy_image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf5\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00IEND\xaeB`\x82'
        
        files = {'files': ('test_image.png', dummy_image_data, 'image/png')}
        self.client.post("/api/upload", files=files)
    
    @task(1)
    def trigger_retrain(self):
        """Trigger model retraining"""
        self.client.post("/api/retrain")
    
    @task(5)
    def predict_image(self):
        """Make prediction on uploaded image - highest frequency"""
        # Create a dummy image file for testing
        dummy_image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf5\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00IEND\xaeB`\x82'
        
        files = {'file': ('test_prediction.png', dummy_image_data, 'image/png')}
        self.client.post("/api/predict", files=files)

class PredictionUser(HttpUser):
    """User class focused on prediction requests"""
    wait_time = between(0.5, 2)  # Faster requests for prediction testing
    
    @task(8)
    def predict_image(self):
        """High-frequency prediction requests"""
        dummy_image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\x0cIDATx\x9cc```\x00\x00\x00\x04\x00\x01\xf5\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00IEND\xaeB`\x82'
        
        files = {'file': ('test_prediction.png', dummy_image_data, 'image/png')}
        self.client.post("/api/predict", files=files)
    
    @task(2)
    def health_check(self):
        """Occasional health checks"""
        self.client.get("/api/health")

class MonitoringUser(HttpUser):
    """User class focused on monitoring requests"""
    wait_time = between(2, 5)  # Slower requests for monitoring
    
    @task(3)
    def get_statistics(self):
        """Get system statistics"""
        self.client.get("/api/statistics")
    
    @task(2)
    def get_predictions(self):
        """Get recent predictions"""
        self.client.get("/api/predictions")
    
    @task(1)
    def get_model_info(self):
        """Get model information"""
        self.client.get("/api/model-info")
    
    @task(1)
    def health_check(self):
        """Health check"""
        self.client.get("/api/health")

# Configuration for different load testing scenarios
class Config:
    """Configuration for different load testing scenarios"""
    
    @staticmethod
    def light_load():
        """Light load testing - 10 users, 1-3 seconds between requests"""
        return {
            'users': 10,
            'spawn_rate': 1,
            'run_time': '2m'
        }
    
    @staticmethod
    def medium_load():
        """Medium load testing - 50 users, 1-3 seconds between requests"""
        return {
            'users': 50,
            'spawn_rate': 5,
            'run_time': '5m'
        }
    
    @staticmethod
    def heavy_load():
        """Heavy load testing - 100 users, 1-3 seconds between requests"""
        return {
            'users': 100,
            'spawn_rate': 10,
            'run_time': '10m'
        }
    
    @staticmethod
    def stress_test():
        """Stress testing - 200 users, 0.5-2 seconds between requests"""
        return {
            'users': 200,
            'spawn_rate': 20,
            'run_time': '15m'
        }
    
    @staticmethod
    def prediction_focus():
        """Focus on prediction endpoints - 80% prediction requests"""
        return {
            'users': 75,
            'spawn_rate': 8,
            'run_time': '8m'
        }

# Usage instructions
"""
To run load tests with different scenarios:

1. Light Load Test:
   locust -f locustfile.py --host=http://localhost:5000 --users=10 --spawn-rate=1 --run-time=2m

2. Medium Load Test:
   locust -f locustfile.py --host=http://localhost:5000 --users=50 --spawn-rate=5 --run-time=5m

3. Heavy Load Test:
   locust -f locustfile.py --host=http://localhost:5000 --users=100 --spawn-rate=10 --run-time=10m

4. Stress Test:
   locust -f locustfile.py --host=http://localhost:5000 --users=200 --spawn-rate=20 --run-time=15m

5. Prediction Focus Test:
   locust -f locustfile.py --host=http://localhost:5000 --users=75 --spawn-rate=8 --run-time=8m

Or use the web interface:
   locust -f locustfile.py --host=http://localhost:5000
   Then open http://localhost:8089 in your browser
""" 