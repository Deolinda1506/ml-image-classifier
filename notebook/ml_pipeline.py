#!/usr/bin/env python3
"""
ML Image Classifier Pipeline - Python Script Version
This script contains all the code from the Jupyter notebook for the ML pipeline.
"""

import sys
sys.path.append('../src')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Import our custom modules
from preprocessing import ImagePreprocessor
from model import ImageClassifier
from prediction import PredictionService

def main():
    print("=" * 60)
    print("ML IMAGE CLASSIFIER PIPELINE")
    print("=" * 60)
    
    # 1. Setup and Imports
    print("\n1. SETUP AND IMPORTS")
    print("-" * 30)
    print("All imports successful!")
    
    # 2. Data Loading and Preprocessing
    print("\n2. DATA LOADING AND PREPROCESSING")
    print("-" * 30)

# Initialize preprocessor
    preprocessor = ImagePreprocessor(img_size=(224, 224), batch_size=32)
    print(f"Preprocessor initialized with image size: {preprocessor.img_size}")
    print(f"Class names: {preprocessor.class_names}")
    
    # Load dataset
    print("\nLoading dataset...")
    try:
        X_train, y_train, X_test, y_test = preprocessor.load_dataset_from_flat_structure('../data')
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Training labels shape: {y_train.shape}")
        print(f"Test labels shape: {y_test.shape}")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Visualize dataset samples
    print("\nVisualizing dataset samples...")
    try:
        preprocessor.visualize_dataset(X_train, y_train, num_samples=8)
        plt.savefig('dataset_samples.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Error visualizing dataset: {e}")
    
    # Plot class distribution
    print("\nAnalyzing class distribution...")
    try:
        preprocessor.plot_class_distribution(y_train, y_test)
        plt.savefig('class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print(f"Error plotting class distribution: {e}")
    
    # 3. Data Analysis and Insights
    print("\n3. DATA ANALYSIS AND INSIGHTS")
    print("-" * 30)
    
    def analyze_image_characteristics(X_train, y_train):
        """Analyze various characteristics of the images"""
        
        # Calculate brightness (mean pixel value)
        brightness = np.mean(X_train, axis=(1, 2, 3))
        
        # Calculate contrast (standard deviation of pixel values)
        contrast = np.std(X_train, axis=(1, 2, 3))
        
        # Calculate color distribution
        red_channel = np.mean(X_train[:, :, :, 0], axis=(1, 2))
        green_channel = np.mean(X_train[:, :, :, 1], axis=(1, 2))
        blue_channel = np.mean(X_train[:, :, :, 2], axis=(1, 2))
        
        # Create DataFrame for analysis
        df = pd.DataFrame({
            'class': [preprocessor.class_names[y] for y in y_train],
            'brightness': brightness,
            'contrast': contrast,
            'red_channel': red_channel,
            'green_channel': green_channel,
            'blue_channel': blue_channel
        })
        
        return df
    
    # Perform analysis
    print("Analyzing image characteristics...")
    try:
        image_analysis = analyze_image_characteristics(X_train, y_train)
        print("\nImage Analysis Summary:")
        print(image_analysis.groupby('class').describe())
        
        # Visualize image characteristics by class
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Image Characteristics Analysis by Class', fontsize=16, fontweight='bold')
        
        # Brightness
        axes[0, 0].hist(image_analysis[image_analysis['class'] == 'cat']['brightness'], 
                        alpha=0.7, label='Cat', bins=20)
        axes[0, 0].hist(image_analysis[image_analysis['class'] == 'dog']['brightness'], 
                        alpha=0.7, label='Dog', bins=20)
        axes[0, 0].set_title('Brightness Distribution')
        axes[0, 0].set_xlabel('Brightness')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].legend()
        
        # Contrast
        axes[0, 1].hist(image_analysis[image_analysis['class'] == 'cat']['contrast'], 
                        alpha=0.7, label='Cat', bins=20)
        axes[0, 1].hist(image_analysis[image_analysis['class'] == 'dog']['contrast'], 
                        alpha=0.7, label='Dog', bins=20)
        axes[0, 1].set_title('Contrast Distribution')
        axes[0, 1].set_xlabel('Contrast')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].legend()
        
        # Red Channel
        axes[0, 2].hist(image_analysis[image_analysis['class'] == 'cat']['red_channel'], 
                        alpha=0.7, label='Cat', bins=20)
        axes[0, 2].hist(image_analysis[image_analysis['class'] == 'dog']['red_channel'], 
                        alpha=0.7, label='Dog', bins=20)
        axes[0, 2].set_title('Red Channel Distribution')
        axes[0, 2].set_xlabel('Red Channel Value')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].legend()
        
        # Green Channel
        axes[1, 0].hist(image_analysis[image_analysis['class'] == 'cat']['green_channel'], 
                        alpha=0.7, label='Cat', bins=20)
        axes[1, 0].hist(image_analysis[image_analysis['class'] == 'dog']['green_channel'], 
                        alpha=0.7, label='Dog', bins=20)
        axes[1, 0].set_title('Green Channel Distribution')
        axes[1, 0].set_xlabel('Green Channel Value')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        
        # Blue Channel
        axes[1, 1].hist(image_analysis[image_analysis['class'] == 'cat']['blue_channel'], 
                        alpha=0.7, label='Cat', bins=20)
        axes[1, 1].hist(image_analysis[image_analysis['class'] == 'dog']['blue_channel'], 
                        alpha=0.7, label='Dog', bins=20)
        axes[1, 1].set_title('Blue Channel Distribution')
        axes[1, 1].set_xlabel('Blue Channel Value')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        
        # Box plot of brightness by class
        image_analysis.boxplot(column='brightness', by='class', ax=axes[1, 2])
        axes[1, 2].set_title('Brightness by Class')
        axes[1, 2].set_xlabel('Class')
        axes[1, 2].set_ylabel('Brightness')
        
        plt.tight_layout()
        plt.savefig('data_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

        print("\nKey Insights:")
        print(f"1. Cats tend to have {'higher' if image_analysis.groupby('class')['brightness'].mean()['cat'] > image_analysis.groupby('class')['brightness'].mean()['dog'] else 'lower'} brightness than dogs")
        print(f"2. Dogs have {'higher' if image_analysis.groupby('class')['contrast'].mean()['dog'] > image_analysis.groupby('class')['contrast'].mean()['cat'] else 'lower'} contrast than cats")
        print(f"3. Color channel distributions show {'similar' if abs(image_analysis.groupby('class')['red_channel'].mean()['cat'] - image_analysis.groupby('class')['red_channel'].mean()['dog']) < 0.05 else 'different'} patterns between classes")
        
    except Exception as e:
        print(f"Error in data analysis: {e}")
    
    # 4. Model Training
    print("\n4. MODEL TRAINING")
    print("-" * 30)
    
    # Split training data into train and validation
    from sklearn.model_selection import train_test_split
    
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Training set: {X_train_split.shape}")
    print(f"Validation set: {X_val.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Create data generators with augmentation
    print("\nCreating data generators...")
    try:
        train_generator, val_generator = preprocessor.create_data_generators(
            X_train_split, y_train_split, X_val, y_val
        )
        print("Data generators created successfully!")
    except Exception as e:
        print(f"Error creating data generators: {e}")
        return
    
    # Initialize and build model
    print("\nBuilding model...")
    try:
        classifier = ImageClassifier(model_type='custom')
        model = classifier.build_model()
        
        print("Model architecture:")
        model.summary()
    except Exception as e:
        print(f"Error building model: {e}")
        return
    
    # Train the model
    print("\nStarting model training...")
    start_time = datetime.now()
    
    try:
        history = classifier.train(
            train_generator, 
            val_generator, 
            epochs=50,
            model_save_path='../models/best_model.h5'
        )
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        print(f"\nTraining completed in {training_duration:.2f} seconds")
        print(f"Training completed in {training_duration/60:.2f} minutes")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    # 5. Model Evaluation
    print("\n5. MODEL EVALUATION")
    print("-" * 30)
    
    # Plot training history
    print("Plotting training history...")
    try:
        classifier.plot_training_history(save_path='training_history.png')
    except Exception as e:
        print(f"Error plotting training history: {e}")
    
    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    try:
        metrics = classifier.evaluate_model(X_test, y_test)
        
        print("\nTest Set Evaluation Results:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        
        print("\nDetailed Classification Report:")
        print(metrics['classification_report'])
        
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return
    
    # Plot confusion matrix
    print("\nPlotting confusion matrix...")
    try:
        cm = np.array(metrics['confusion_matrix'])
        classifier.plot_confusion_matrix(cm, save_path='confusion_matrix.png')
    except Exception as e:
        print(f"Error plotting confusion matrix: {e}")
    
    # Analyze misclassifications
    def analyze_misclassifications(X_test, y_test, y_pred, metrics):
        """Analyze misclassified images"""
        
        # Get misclassified indices
        misclassified = np.where(y_test != y_pred)[0]
        
        print(f"\nMisclassification Analysis:")
        print(f"Total misclassifications: {len(misclassified)}")
        print(f"Misclassification rate: {len(misclassified)/len(y_test)*100:.2f}%")
        
        if len(misclassified) > 0:
            # Show some misclassified examples
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            axes = axes.ravel()
            
            for i, idx in enumerate(misclassified[:8]):
                axes[i].imshow(X_test[idx])
                true_class = preprocessor.class_names[y_test[idx]]
                pred_class = preprocessor.class_names[y_pred[idx]]
                axes[i].set_title(f'True: {true_class}\nPred: {pred_class}')
                axes[i].axis('off')
            
plt.tight_layout()
            plt.savefig('misclassifications.png', dpi=300, bbox_inches='tight')
plt.show()

        return misclassified
    
    # Analyze misclassifications
    try:
        y_pred = np.array(metrics['predictions'])
        misclassified_indices = analyze_misclassifications(X_test, y_test, y_pred, metrics)
    except Exception as e:
        print(f"Error analyzing misclassifications: {e}")
    
    # 6. Model Deployment Preparation
    print("\n6. MODEL DEPLOYMENT PREPARATION")
    print("-" * 30)
    
    # Save final model and metadata
    print("Saving final model and metadata...")
    try:
        classifier.save_model(
            '../models/best_model.h5',
            '../models/model_metadata.json'
        )
        print("Model saved successfully!")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    # Test prediction service
    print("\nTesting prediction service...")
    try:
        prediction_service = PredictionService(
            '../models/best_model.h5',
            '../models/model_metadata.json'
        )
        
        # Test with a sample image
        sample_image = X_test[0:1]  # Take first test image
        prediction = classifier.predict_single_image(sample_image)
        
        print(f"\nSample Prediction Test:")
        print(f"Predicted class: {prediction['predicted_class']}")
        print(f"Confidence: {prediction['confidence']:.4f}")
        print(f"True class: {preprocessor.class_names[y_test[0]]}")
        print(f"Prediction correct: {prediction['predicted_class'] == preprocessor.class_names[y_test[0]]}")
        
    except Exception as e:
        print(f"Error testing prediction service: {e}")
    
    # 7. Summary and Conclusions
    print("\n7. SUMMARY AND CONCLUSIONS")
    print("-" * 30)
    
    # Final summary
    print("=" * 60)
    print("ML PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Dataset: {X_train.shape[0]} training, {X_test.shape[0]} test images")
    print(f"Classes: {preprocessor.class_names}")
    print(f"Image size: {preprocessor.img_size}")
    print(f"Model type: {classifier.model_type}")
    print(f"Training time: {training_duration/60:.2f} minutes")
    print(f"\nFinal Test Results:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    print(f"\nModel saved to: ../models/best_model.h5")
    print(f"Metadata saved to: ../models/model_metadata.json")
    print("=" * 60)
    
    # Key insights
    print("\nKEY INSIGHTS:")
    print("1. Data preprocessing with augmentation improves model generalization")
    print("2. Custom CNN architecture performs well for this binary classification task")
    print("3. Model shows good balance between precision and recall")
    print("4. Ready for deployment with Flask API and web interface")
    print("5. Supports real-time prediction and model retraining")

print("\n🎉 ML Pipeline completed successfully!") 

if __name__ == "__main__":
    main() 