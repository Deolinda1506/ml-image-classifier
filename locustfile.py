from locust import HttpUser, task, between
import random
import os

class MLAPIUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Initialize user session"""
        # Check if model is loaded
        response = self.client.get("/status")
        if response.status_code != 200:
            print("Warning: Model not loaded or API not responding")
    
    @task(3)
    def health_check(self):
        """Health check endpoint"""
        self.client.get("/health")
    
    @task(2)
    def get_status(self):
        """Get model status"""
        self.client.get("/status")
    
    @task(1)
    def predict_image(self):
        """Make prediction on a random test image"""
        # Get list of test images
        test_dir = "data/test"
        if os.path.exists(test_dir):
            test_images = [f for f in os.listdir(test_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if test_images:
                # Select random image
                image_file = random.choice(test_images)
                image_path = os.path.join(test_dir, image_file)
                
                # Upload image for prediction
                with open(image_path, 'rb') as f:
                    files = {'image': (image_file, f, 'image/jpeg')}
                    self.client.post("/predict", files=files)
    
    @task(1)
    def get_training_status(self):
        """Get training status"""
        self.client.get("/training-status")
    
    @task(1)
    def start_training(self):
        """Start model training (low frequency)"""
        if random.random() < 0.1:  # Only 10% of the time
            self.client.post("/train")
    
    @task(1)
    def upload_data(self):
        """Upload data for retraining (low frequency)"""
        if random.random() < 0.05:  # Only 5% of the time
            # Get a random image to upload
            test_dir = "data/test"
            if os.path.exists(test_dir):
                test_images = [f for f in os.listdir(test_dir) 
                              if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if test_images:
                    image_file = random.choice(test_images)
                    image_path = os.path.join(test_dir, image_file)
                    
                    with open(image_path, 'rb') as f:
                        files = {'files[]': (image_file, f, 'image/jpeg')}
                        self.client.post("/upload-data", files=files)

class HighLoadUser(HttpUser):
    """User class for high load testing"""
    wait_time = between(0.1, 0.5)  # Very fast requests
    
    @task(5)
    def rapid_predictions(self):
        """Make rapid predictions"""
        test_dir = "data/test"
        if os.path.exists(test_dir):
            test_images = [f for f in os.listdir(test_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if test_images:
                image_file = random.choice(test_images)
                image_path = os.path.join(test_dir, image_file)
                
                with open(image_path, 'rb') as f:
                    files = {'image': (image_file, f, 'image/jpeg')}
                    self.client.post("/predict", files=files)
    
    @task(2)
    def rapid_health_checks(self):
        """Rapid health checks"""
        self.client.get("/health")

class StressTestUser(HttpUser):
    """User class for stress testing"""
    wait_time = between(0.05, 0.2)  # Very rapid requests
    
    @task(10)
    def stress_predictions(self):
        """Stress test predictions"""
        test_dir = "data/test"
        if os.path.exists(test_dir):
            test_images = [f for f in os.listdir(test_dir) 
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            if test_images:
                image_file = random.choice(test_images)
                image_path = os.path.join(test_dir, image_file)
                
                with open(image_path, 'rb') as f:
                    files = {'image': (image_file, f, 'image/jpeg')}
                    self.client.post("/predict", files=files)
    
    @task(5)
    def stress_health_checks(self):
        """Stress test health checks"""
        self.client.get("/health")
    
    @task(2)
    def stress_status_checks(self):
        """Stress test status checks"""
        self.client.get("/status") 