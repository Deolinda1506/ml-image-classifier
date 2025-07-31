import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import cv2
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class ImageClassifier:
    def __init__(self, model_type='random_forest'):
        """
        Initialize the image classifier
        
        Args:
            model_type (str): Type of model to use ('random_forest', 'svm', 'knn')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.class_names = ['cat', 'dog']
        self.training_history = []
        
    def extract_features(self, images):
        """
        Extract features from images using color histograms and texture features
        
        Args:
            images (np.array): Array of images
            
        Returns:
            np.array: Feature matrix
        """
        features = []
        
        for img in images:
            # Convert to RGB if needed
            if len(img.shape) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
                
            # Color histogram features
            hist_r = cv2.calcHist([img_rgb], [0], None, [256], [0, 256]).flatten()
            hist_g = cv2.calcHist([img_rgb], [1], None, [256], [0, 256]).flatten()
            hist_b = cv2.calcHist([img_rgb], [2], None, [256], [0, 256]).flatten()
            
            # Normalize histograms
            hist_r = hist_r / np.sum(hist_r)
            hist_g = hist_g / np.sum(hist_g)
            hist_b = hist_b / np.sum(hist_b)
            
            # Texture features (GLCM-like)
            gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
            
            # Edge features
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # Local Binary Pattern (simplified)
            lbp_features = self._compute_lbp(gray)
            
            # Combine all features
            feature_vector = np.concatenate([
                hist_r, hist_g, hist_b,  # Color features (768)
                [edge_density],          # Edge density (1)
                lbp_features             # LBP features (256)
            ])
            
            features.append(feature_vector)
            
        return np.array(features)
    
    def _compute_lbp(self, gray_img):
        """
        Compute Local Binary Pattern histogram
        """
        # Simplified LBP implementation
        height, width = gray_img.shape
        lbp = np.zeros((height, width), dtype=np.uint8)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                center = gray_img[i, j]
                code = 0
                # 8-neighbor LBP
                neighbors = [
                    gray_img[i-1, j-1], gray_img[i-1, j], gray_img[i-1, j+1],
                    gray_img[i, j+1], gray_img[i+1, j+1], gray_img[i+1, j],
                    gray_img[i+1, j-1], gray_img[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp[i, j] = code
        
        # Compute histogram
        hist, _ = np.histogram(lbp.flatten(), bins=256, range=(0, 256))
        return hist / np.sum(hist)
    
    def create_model(self):
        """
        Create the specified model
        """
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'svm':
            self.model = SVC(
                kernel='rbf',
                C=1.0,
                random_state=42,
                probability=True
            )
        elif self.model_type == 'knn':
            self.model = KNeighborsClassifier(
                n_neighbors=5,
                weights='uniform'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the model
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data (optional)
        """
        print(f"Training {self.model_type} model...")
        
        # Extract features
        print("Extracting features from training images...")
        X_train_features = self.extract_features(X_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train_features)
        
        # Create and train model
        self.create_model()
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate on training set
        train_pred = self.model.predict(X_train_scaled)
        train_accuracy = accuracy_score(y_train, train_pred)
        
        print(f"Training accuracy: {train_accuracy:.4f}")
        
        # Evaluate on validation set if provided
        if X_val is not None and y_val is not None:
            print("Extracting features from validation images...")
            X_val_features = self.extract_features(X_val)
            X_val_scaled = self.scaler.transform(X_val_features)
            
            val_pred = self.model.predict(X_val_scaled)
            val_accuracy = accuracy_score(y_val, val_pred)
            print(f"Validation accuracy: {val_accuracy:.4f}")
            
            # Store training history
            self.training_history.append({
                'epoch': 1,
                'train_accuracy': train_accuracy,
                'val_accuracy': val_accuracy,
                'timestamp': datetime.now().isoformat()
            })
        
        print("Training completed!")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Args:
            X_test, y_test: Test data
            
        Returns:
            dict: Evaluation metrics
        """
        print("Evaluating model...")
        
        # Extract features
        X_test_features = self.extract_features(X_test)
        X_test_scaled = self.scaler.transform(X_test_features)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        
        return metrics
    
    def predict(self, image):
        """
        Make prediction on a single image
        
        Args:
            image: Input image
            
        Returns:
            tuple: (prediction, probability)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Extract features
        features = self.extract_features([image])
        features_scaled = self.scaler.transform(features)
        
        # Make prediction
        prediction = self.model.predict(features_scaled)[0]
        probability = self.model.predict_proba(features_scaled)[0]
        
        return prediction, probability
    
    def plot_confusion_matrix(self, cm, save_path=None):
        """
        Plot confusion matrix
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_classification_report(self, y_true, y_pred, save_path=None):
        """
        Plot classification report
        """
        report = classification_report(y_true, y_pred, 
                                     target_names=self.class_names,
                                     output_dict=True)
        
        # Convert to DataFrame for easier plotting
        report_df = pd.DataFrame(report).transpose()
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(report_df.iloc[:-1, :].astype(float), 
                   annot=True, cmap='YlOrRd', fmt='.3f')
        plt.title('Classification Report')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath):
        """
        Save the trained model
        """
        if self.model is None:
            raise ValueError("No model to save!")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'model_type': self.model_type,
            'class_names': self.class_names,
            'training_history': self.training_history
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load a trained model
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.model_type = model_data['model_type']
        self.class_names = model_data['class_names']
        self.training_history = model_data.get('training_history', [])
        
        print(f"Model loaded from {filepath}")
    
    def get_model_info(self):
        """
        Get information about the model
        """
        if self.model is None:
            return {"status": "No model trained"}
        
        info = {
            "model_type": self.model_type,
            "class_names": self.class_names,
            "is_trained": True,
            "training_history": self.training_history
        }
        
        if hasattr(self.model, 'n_estimators'):
            info["n_estimators"] = self.model.n_estimators
        if hasattr(self.model, 'n_neighbors'):
            info["n_neighbors"] = self.model.n_neighbors
            
        return info 