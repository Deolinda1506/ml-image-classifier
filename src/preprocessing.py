import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class ImagePreprocessor:
    def __init__(self, img_size=(224, 224), batch_size=32):
        """
        Initialize the image preprocessor
        
        Args:
            img_size (tuple): Target image size (width, height)
            batch_size (int): Batch size for data loading
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = ['cat', 'dog']
        
    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess a single image
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            np.array: Preprocessed image
        """
        # Load image using OpenCV
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, self.img_size)
        
        return img
    
    def load_dataset(self, data_dir):
        """
        Load dataset from organized directory structure
        
        Args:
            data_dir (str): Path to data directory
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        X_train, y_train = [], []
        X_test, y_test = [], []
        
        # Load training data
        train_dir = os.path.join(data_dir, 'train')
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(train_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        try:
                            img = self.load_and_preprocess_image(img_path)
                            X_train.append(img)
                            y_train.append(class_idx)
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
        
        # Load test data
        test_dir = os.path.join(data_dir, 'test')
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(test_dir, class_name)
            if os.path.exists(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_name)
                        try:
                            img = self.load_and_preprocess_image(img_path)
                            X_test.append(img)
                            y_test.append(class_idx)
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        return X_train, y_train, X_test, y_test
    
    def load_dataset_from_flat_structure(self, data_dir):
        """
        Load dataset from flat directory structure (cat.0.jpg, dog.0.jpg, etc.)
        
        Args:
            data_dir (str): Path to data directory
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        X_train, y_train = [], []
        X_test, y_test = [], []
        
        # Load training data
        train_dir = os.path.join(data_dir, 'train')
        for img_name in os.listdir(train_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(train_dir, img_name)
                try:
                    img = self.load_and_preprocess_image(img_path)
                    X_train.append(img)
                    # Extract class from filename (cat.0.jpg -> 0, dog.0.jpg -> 1)
                    class_name = img_name.split('.')[0]
                    class_idx = self.class_names.index(class_name)
                    y_train.append(class_idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        # Load test data
        test_dir = os.path.join(data_dir, 'test')
        for img_name in os.listdir(test_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(test_dir, img_name)
                try:
                    img = self.load_and_preprocess_image(img_path)
                    X_test.append(img)
                    # Extract class from filename
                    class_name = img_name.split('.')[0]
                    class_idx = self.class_names.index(class_name)
                    y_test.append(class_idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
        
        # Convert to numpy arrays
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        X_test = np.array(X_test)
        y_test = np.array(y_test)
        
        return X_train, y_train, X_test, y_test
    
    def create_data_generators(self, X_train, y_train, X_val, y_val):
        """
        Create data generators with augmentation for training
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            
        Returns:
            tuple: (train_generator, val_generator)
        """
        # For scikit-learn, we'll return the data as is
        # Augmentation will be handled in the model training if needed
        return (X_train, y_train), (X_val, y_val)
    
    def augment_image(self, image):
        """
        Apply data augmentation to a single image
        
        Args:
            image (np.array): Input image
            
        Returns:
            np.array: Augmented image
        """
        # Horizontal flip
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
        
        # Rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-15, 15)
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            image = cv2.warpAffine(image, rotation_matrix, (width, height))
        
        # Brightness adjustment
        if np.random.random() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=0)
        
        return image
    
    def visualize_dataset(self, X, y, num_samples=8):
        """
        Visualize random samples from the dataset
        
        Args:
            X (np.array): Images
            y (np.array): Labels
            num_samples (int): Number of samples to visualize
        """
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for i in range(num_samples):
            idx = np.random.randint(0, len(X))
            axes[i].imshow(X[idx])
            axes[i].set_title(f'{self.class_names[y[idx]]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_class_distribution(self, y_train, y_test):
        """
        Plot class distribution in training and test sets
        
        Args:
            y_train, y_test: Training and test labels
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Training set distribution
        train_counts = [np.sum(y_train == i) for i in range(len(self.class_names))]
        ax1.bar(self.class_names, train_counts, color=['orange', 'blue'])
        ax1.set_title('Training Set Class Distribution')
        ax1.set_ylabel('Count')
        
        # Test set distribution
        test_counts = [np.sum(y_test == i) for i in range(len(self.class_names))]
        ax2.bar(self.class_names, test_counts, color=['orange', 'blue'])
        ax2.set_title('Test Set Class Distribution')
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Training set: {dict(zip(self.class_names, train_counts))}")
        print(f"Test set: {dict(zip(self.class_names, test_counts))}")
    
    def preprocess_uploaded_image(self, image_file):
        """
        Preprocess an uploaded image file
        
        Args:
            image_file: Uploaded file object
            
        Returns:
            np.array: Preprocessed image
        """
        # Read image from file
        img_array = np.frombuffer(image_file.read(), np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode uploaded image")
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, self.img_size)
        
        return img 