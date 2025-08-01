import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class ImagePreprocessor:
    def __init__(self, img_size=(128, 128), batch_size=32):
        """
        Initialize the image preprocessor with advanced features for TensorFlow
        
        Args:
            img_size (tuple): Target image size (width, height)
            batch_size (int): Batch size for processing
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = ['cat', 'dog']
        
        # Advanced preprocessing parameters for TensorFlow
        self.augmentation_params = {
            'rotation_range': 20,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'horizontal_flip': True,
            'brightness_range': [0.8, 1.2],
            'zoom_range': 0.1,
            'fill_mode': 'nearest'
        }
        
        # Normalization parameters
        self.normalization_type = 'tensorflow'  # 'tensorflow', 'standard', 'minmax'
        
    def load_dataset_from_flat_structure(self, data_dir):
        """
        Load dataset from flat directory structure with advanced preprocessing for TensorFlow
        
        Args:
            data_dir (str): Path to data directory
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        print("Loading dataset with TensorFlow-compatible preprocessing...")
        
        train_dir = os.path.join(data_dir, 'train')
        test_dir = os.path.join(data_dir, 'test')
        
        # Load training data
        X_train, y_train = self._load_images_from_directory(train_dir)
        
        # Load test data
        X_test, y_test = self._load_images_from_directory(test_dir)
        
        # Apply TensorFlow-compatible preprocessing
        X_train = self._apply_tensorflow_preprocessing(X_train)
        X_test = self._apply_tensorflow_preprocessing(X_test)
        
        # Apply data augmentation to training set
        X_train_aug, y_train_aug = self._apply_data_augmentation(X_train, y_train)
        
        print(f"Dataset loaded successfully!")
        print(f"Training set: {X_train_aug.shape} (original: {X_train.shape})")
        print(f"Test set: {X_test.shape}")
        
        return X_train_aug, y_train_aug, X_test, y_test
    
    def _load_images_from_directory(self, directory):
        """Load images from directory with error handling"""
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(directory, class_name)
            if not os.path.exists(class_dir):
                # Try alternative naming
                if class_name == 'cat':
                    class_dir = os.path.join(directory, 'cats')
                elif class_name == 'dog':
                    class_dir = os.path.join(directory, 'dogs')
            
            if os.path.exists(class_dir):
                for filename in os.listdir(class_dir):
                    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                        try:
                            img_path = os.path.join(class_dir, filename)
                            img = self._load_and_preprocess_image(img_path)
                            if img is not None:
                                images.append(img)
                                labels.append(class_idx)
                        except Exception as e:
                            print(f"Error loading {filename}: {e}")
                            continue
        
        return np.array(images), np.array(labels)
    
    def _load_and_preprocess_image(self, image_path):
        """Load and preprocess a single image for TensorFlow"""
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            # Convert BGR to RGB (TensorFlow expects RGB)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, self.img_size)
            
            # Apply noise reduction
            img = cv2.medianBlur(img, 3)
            
            # Apply histogram equalization for better contrast
            img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
            img_yuv[:,:,0] = cv2.equalizeHist(img_yuv[:,:,0])
            img = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
            
            return img
            
        except Exception as e:
            print(f"Error preprocessing image {image_path}: {e}")
            return None
    
    def _apply_tensorflow_preprocessing(self, images):
        """Apply TensorFlow-compatible preprocessing techniques"""
        processed_images = []
        
        for img in images:
            # 1. Normalization for TensorFlow (0-1 range)
            img_normalized = img.astype(np.float32) / 255.0
            
            # 2. Color space transformations
            # Convert to different color spaces for additional features
            img_hsv = cv2.cvtColor(img_normalized.astype(np.float32), cv2.COLOR_RGB2HSV)
            img_lab = cv2.cvtColor(img_normalized.astype(np.float32), cv2.COLOR_RGB2LAB)
            
            # 3. Edge enhancement
            gray = cv2.cvtColor(img_normalized.astype(np.float32), cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            # 4. Apply Gaussian blur for noise reduction
            img_blurred = cv2.GaussianBlur(img_normalized.astype(np.float32), (3, 3), 0)
            
            # 5. Apply unsharp masking for sharpening
            gaussian = cv2.GaussianBlur(img_blurred, (0, 0), 2.0)
            img_sharpened = cv2.addWeighted(img_blurred, 1.5, gaussian, -0.5, 0)
            
            # For TensorFlow, we'll use the sharpened version
            processed_images.append(img_sharpened)
        
        return np.array(processed_images)
    
    def _apply_data_augmentation(self, images, labels):
        """Apply data augmentation techniques for TensorFlow"""
        augmented_images = []
        augmented_labels = []
        
        # Add original images
        augmented_images.extend(images)
        augmented_labels.extend(labels)
        
        # Apply augmentation to each image
        for img, label in zip(images, labels):
            # Generate multiple augmented versions
            for _ in range(2):  # Create 2 augmented versions per image
                aug_img = self._augment_single_image(img)
                augmented_images.append(aug_img)
                augmented_labels.append(label)
        
        return np.array(augmented_images), np.array(augmented_labels)
    
    def _augment_single_image(self, image):
        """Apply augmentation to a single image for TensorFlow"""
        img = image.copy()
        
        # 1. Random rotation
        if np.random.random() > 0.5:
            angle = np.random.uniform(-self.augmentation_params['rotation_range'], 
                                    self.augmentation_params['rotation_range'])
            height, width = img.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, rotation_matrix, (width, height))
        
        # 2. Random horizontal flip
        if self.augmentation_params['horizontal_flip'] and np.random.random() > 0.5:
            img = cv2.flip(img, 1)
        
        # 3. Random brightness adjustment
        if np.random.random() > 0.5:
            alpha = np.random.uniform(*self.augmentation_params['brightness_range'])
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
        
        # 4. Random zoom
        if np.random.random() > 0.5:
            zoom_factor = np.random.uniform(1 - self.augmentation_params['zoom_range'], 
                                           1 + self.augmentation_params['zoom_range'])
            height, width = img.shape[:2]
            new_height, new_width = int(height * zoom_factor), int(width * zoom_factor)
            img = cv2.resize(img, (new_width, new_height))
            img = cv2.resize(img, (width, height))
        
        # 5. Random noise addition
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 0.05, img.shape)
            img = np.clip(img + noise, 0, 1)
        
        return img
    
    def preprocess_uploaded_image(self, image_file):
        """Preprocess uploaded image file for TensorFlow"""
        try:
            # Read image from file
            img_array = np.frombuffer(image_file.read(), np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            
            if img is None:
                raise ValueError("Could not decode image")
            
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Resize image
            img = cv2.resize(img, self.img_size)
            
            # Apply TensorFlow preprocessing
            img = self._apply_tensorflow_preprocessing([img])[0]
            
            # Reshape for prediction
            img = img.reshape(1, *img.shape)
            
            return img
            
        except Exception as e:
            print(f"Error preprocessing uploaded image: {e}")
            raise
    
    def visualize_dataset(self, images, labels, num_samples=8):
        """Visualize dataset samples with advanced analysis for TensorFlow"""
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.ravel()
        
        for i in range(min(num_samples, len(images))):
            # Convert back to 0-255 range for visualization
            img_display = (images[i] * 255).astype(np.uint8)
            axes[i].imshow(img_display)
            class_name = self.class_names[labels[i]]
            axes[i].set_title(f'{class_name}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        # Additional visualizations
        self._plot_image_statistics(images, labels)
    
    def _plot_image_statistics(self, images, labels):
        """Plot advanced image statistics for TensorFlow data"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Convert to 0-255 range for statistics
        images_255 = (images * 255).astype(np.uint8)
        
        # 1. Brightness distribution
        brightness = np.mean(images_255, axis=(1, 2, 3))
        for class_idx, class_name in enumerate(self.class_names):
            class_brightness = brightness[labels == class_idx]
            axes[0, 0].hist(class_brightness, alpha=0.7, label=class_name, bins=20)
        axes[0, 0].set_xlabel('Brightness')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Brightness Distribution by Class')
        axes[0, 0].legend()
        
        # 2. Contrast distribution
        contrast = np.std(images_255, axis=(1, 2, 3))
        for class_idx, class_name in enumerate(self.class_names):
            class_contrast = contrast[labels == class_idx]
            axes[0, 1].hist(class_contrast, alpha=0.7, label=class_name, bins=20)
        axes[0, 1].set_xlabel('Contrast')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Contrast Distribution by Class')
        axes[0, 1].legend()
        
        # 3. Color channel distributions
        for class_idx, class_name in enumerate(self.class_names):
            class_images = images_255[labels == class_idx]
            if len(class_images) > 0:
                mean_r = np.mean(class_images[:, :, :, 0])
                mean_g = np.mean(class_images[:, :, :, 1])
                mean_b = np.mean(class_images[:, :, :, 2])
                axes[1, 0].scatter([mean_r, mean_g, mean_b], [0, 1, 2], 
                                 label=class_name, alpha=0.7, s=100)
        axes[1, 0].set_xlabel('Mean Channel Value')
        axes[1, 0].set_ylabel('Channel (R=0, G=1, B=2)')
        axes[1, 0].set_title('Mean Color Channel Values by Class')
        axes[1, 0].legend()
        
        # 4. Image size distribution
        image_sizes = [img.shape[0] * img.shape[1] for img in images_255]
        for class_idx, class_name in enumerate(self.class_names):
            class_sizes = [image_sizes[i] for i in range(len(images_255)) if labels[i] == class_idx]
            axes[1, 1].hist(class_sizes, alpha=0.7, label=class_name, bins=20)
        axes[1, 1].set_xlabel('Image Size (pixels)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Image Size Distribution by Class')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.savefig('tensorflow_image_statistics.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_distribution(self, y_train, y_test):
        """Plot class distribution with detailed analysis for TensorFlow"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Training set distribution
        train_counts = [np.sum(y_train == i) for i in range(len(self.class_names))]
        axes[0].bar(self.class_names, train_counts, color=['skyblue', 'lightgreen'])
        axes[0].set_title('Training Set Class Distribution')
        axes[0].set_ylabel('Number of Images')
        for i, count in enumerate(train_counts):
            axes[0].text(i, count + 1, str(count), ha='center', va='bottom')
        
        # Test set distribution
        test_counts = [np.sum(y_test == i) for i in range(len(self.class_names))]
        axes[1].bar(self.class_names, test_counts, color=['lightcoral', 'gold'])
        axes[1].set_title('Test Set Class Distribution')
        axes[1].set_ylabel('Number of Images')
        for i, count in enumerate(test_counts):
            axes[1].text(i, count + 1, str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('tensorflow_class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print statistics
        print("\nDataset Statistics:")
        print(f"Training set: {len(y_train)} images")
        print(f"Test set: {len(y_test)} images")
        print(f"Total: {len(y_train) + len(y_test)} images")
        
        print("\nClass Distribution:")
        for i, class_name in enumerate(self.class_names):
            train_count = np.sum(y_train == i)
            test_count = np.sum(y_test == i)
            print(f"{class_name}: {train_count} train, {test_count} test")
    
    def create_data_generators(self, X_train, y_train, X_val, y_val):
        """Create data generators for TensorFlow training (compatibility method)"""
        # This method is kept for compatibility but returns the data as-is
        # since we're using TensorFlow's ImageDataGenerator
        return (X_train, y_train), (X_val, y_val) 