import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf

class ImagePreprocessor:
    def __init__(self, img_size=(224, 224), batch_size=32):
        """
        Initialize the image preprocessor
        
        Args:
            img_size (tuple): Target image size (width, height)
            batch_size (int): Batch size for training
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = ['cat', 'dog']
        self.num_classes = len(self.class_names)
        
    def load_and_preprocess_image(self, image_path):
        """
        Load and preprocess a single image
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Load image
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize image
        img = cv2.resize(img, self.img_size)
        
        # Normalize pixel values
        img = img.astype(np.float32) / 255.0
        
        return img
    
    def load_dataset(self, data_dir, is_test_set=False):
        """
        Load dataset from flat naming structure
        
        Args:
            data_dir (str): Path to data directory
            is_test_set (bool): Whether this is a test set (different naming)
            
        Returns:
            tuple: (images, labels, file_paths)
        """
        images = []
        labels = []
        file_paths = []
        
        # Get all files in the directory
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(data_dir, filename)
                    
                    if is_test_set:
                        # For test set, we don't have class names in filenames
                        # We'll assign dummy labels (0) for evaluation purposes
                        # In practice, you'd have the true labels separately
                        try:
                            img = self.load_and_preprocess_image(file_path)
                            images.append(img)
                            labels.append(0)  # Dummy label for test set
                            file_paths.append(file_path)
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
                    else:
                        # For training set, extract class name from filename
                        class_name = filename.split('.')[0]
                        
                        # Check if it's a valid class
                        if class_name in self.class_names:
                            try:
                                img = self.load_and_preprocess_image(file_path)
                                images.append(img)
                                labels.append(self.class_names.index(class_name))
                                file_paths.append(file_path)
                            except Exception as e:
                                print(f"Error loading {file_path}: {e}")
        
        return np.array(images), np.array(labels), file_paths
    
    def create_data_generators(self, train_dir, validation_split=0.0, is_test_set=False):
        """
        Create data generators for training (no validation split since we have separate test set)
        
        Args:
            train_dir (str): Path to training data directory
            validation_split (float): Set to 0.0 to use all training data
            is_test_set (bool): Whether this is a test set
            
        Returns:
            tuple: (train_generator, validation_generator)
        """
        # Load all data first
        images, labels, file_paths = self.load_dataset(train_dir, is_test_set=is_test_set)
        
        if len(images) == 0:
            print("Warning: No images found in the dataset!")
            return None, None
        
        # Convert labels to categorical
        from tensorflow.keras.utils import to_categorical
        labels_categorical = to_categorical(labels, num_classes=len(self.class_names))
        
        # Use all data for training (no validation split)
        if validation_split > 0:
            # Split data into train and validation
            from sklearn.model_selection import train_test_split
            X_train, X_val, y_train, y_val = train_test_split(
                images, labels_categorical, test_size=validation_split, 
                random_state=42, stratify=labels
            )
        else:
            # Use all data for training
            X_train, y_train = images, labels_categorical
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        # Only rescaling for validation
        validation_datagen = ImageDataGenerator(
            rescale=1./255
        )
        
        # Create generators using flow method instead of flow_from_directory
        train_generator = train_datagen.flow(
            X_train, y_train,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Add attributes that the model expects
        train_generator.samples = len(X_train)
        train_generator.num_classes = len(self.class_names)
        train_generator.class_indices = {name: i for i, name in enumerate(self.class_names)}
        
        # Create validation generator
        if validation_split > 0:
            # Use actual validation data
            validation_generator = validation_datagen.flow(
                X_val, y_val,
                batch_size=self.batch_size,
                shuffle=False
            )
            validation_generator.samples = len(X_val)
        else:
            # Create a simple validation generator with the same data (for compatibility)
            # In practice, you'll use the separate test set for evaluation
            validation_generator = validation_datagen.flow(
                X_train, y_train,  # Use same data for validation (will be overridden by test set)
                batch_size=self.batch_size,
                shuffle=False
            )
            validation_generator.samples = len(X_train)
        
        validation_generator.num_classes = len(self.class_names)
        validation_generator.class_indices = {name: i for i, name in enumerate(self.class_names)}
        
        return train_generator, validation_generator
    
    def analyze_dataset(self, data_dir):
        """
        Analyze the dataset with flat naming structure
        
        Args:
            data_dir (str): Path to data directory
            
        Returns:
            dict: Dataset statistics
        """
        class_counts = {}
        image_sizes = []
        
        # Initialize class counts
        for class_name in self.class_names:
            class_counts[class_name] = 0
        
        if os.path.exists(data_dir):
            for filename in os.listdir(data_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    # Extract class name from filename (e.g., 'cat.0.jpg' -> 'cat')
                    class_name = filename.split('.')[0]
                    
                    # Count images per class
                    if class_name in self.class_names:
                        class_counts[class_name] += 1
                        
                        # Sample some images to get size distribution
                        if len(image_sizes) < 10:  # Limit to 10 samples
                            file_path = os.path.join(data_dir, filename)
                            try:
                                img = Image.open(file_path)
                                image_sizes.append(img.size)
                            except:
                                pass
        
        # Create visualizations
        self._create_dataset_visualizations(class_counts, image_sizes)
        
        return {
            'class_counts': class_counts,
            'total_images': sum(class_counts.values()),
            'image_sizes': image_sizes
        }
    
    def _create_dataset_visualizations(self, class_counts, image_sizes):
        """
        Create dataset visualizations
        
        Args:
            class_counts (dict): Count of images per class
            image_sizes (list): List of image sizes
        """
        # Create figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # 1. Class distribution
        classes = list(class_counts.keys())
        counts = list(class_counts.values())
        
        axes[0].bar(classes, counts, color=['#FF6B6B', '#4ECDC4'])
        axes[0].set_title('Class Distribution')
        axes[0].set_ylabel('Number of Images')
        axes[0].set_xlabel('Classes')
        
        # Add count labels on bars
        for i, count in enumerate(counts):
            axes[0].text(i, count + 0.5, str(count), ha='center', va='bottom')
        
        # 2. Pie chart
        axes[1].pie(counts, labels=classes, autopct='%1.1f%%', 
                   colors=['#FF6B6B', '#4ECDC4'])
        axes[1].set_title('Class Distribution (Percentage)')
        
        # 3. Image size distribution
        if image_sizes:
            widths = [size[0] for size in image_sizes]
            heights = [size[1] for size in image_sizes]
            
            axes[2].scatter(widths, heights, alpha=0.6, color='#45B7D1')
            axes[2].set_xlabel('Width (pixels)')
            axes[2].set_ylabel('Height (pixels)')
            axes[2].set_title('Image Size Distribution')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def prepare_uploaded_data(self, uploaded_files, labels):
        """
        Prepare uploaded data for retraining
        
        Args:
            uploaded_files (list): List of uploaded file paths
            labels (list): List of corresponding labels
            
        Returns:
            tuple: (images, labels)
        """
        images = []
        processed_labels = []
        
        for file_path, label in zip(uploaded_files, labels):
            try:
                img = self.load_and_preprocess_image(file_path)
                images.append(img)
                processed_labels.append(self.class_names.index(label))
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        
        return np.array(images), np.array(processed_labels) 