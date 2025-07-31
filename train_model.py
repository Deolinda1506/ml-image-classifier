#!/usr/bin/env python3
"""
Standalone script to train the ML model
Usage: python train_model.py [--epochs 50] [--model-type custom]
"""

import argparse
import sys
import os
from datetime import datetime

# Add src to path
sys.path.append('src')

from preprocessing import ImagePreprocessor
from model import ImageClassifier

def train_model(epochs=50, model_type='custom', data_dir='data', model_save_path='models/best_model.h5'):
    """
    Train the ML model
    
    Args:
        epochs (int): Number of training epochs
        model_type (str): Type of model ('custom', 'vgg16', 'resnet50')
        data_dir (str): Path to data directory
        model_save_path (str): Path to save the trained model
    """
    
    print("=" * 60)
    print("ML MODEL TRAINING")
    print("=" * 60)
    print(f"Model Type: {model_type}")
    print(f"Epochs: {epochs}")
    print(f"Data Directory: {data_dir}")
    print(f"Model Save Path: {model_save_path}")
    print("=" * 60)
    
    # Initialize preprocessor
    print("\n1. Initializing preprocessor...")
    preprocessor = ImagePreprocessor(img_size=(224, 224), batch_size=32)
    
    # Load dataset
    print("\n2. Loading dataset...")
    try:
        X_train, y_train, X_test, y_test = preprocessor.load_dataset_from_flat_structure(data_dir)
        print(f"   Training set: {X_train.shape}")
        print(f"   Test set: {X_test.shape}")
        print(f"   Classes: {preprocessor.class_names}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False
    
    # Split training data
    print("\n3. Splitting training data...")
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    print(f"   Training split: {X_train_split.shape}")
    print(f"   Validation split: {X_val.shape}")
    
    # Create data generators
    print("\n4. Creating data generators...")
    train_generator, val_generator = preprocessor.create_data_generators(
        X_train_split, y_train_split, X_val, y_val
    )
    
    # Build model
    print(f"\n5. Building {model_type} model...")
    classifier = ImageClassifier(model_type=model_type)
    model = classifier.build_model()
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    
    # Train model
    print(f"\n6. Training model for {epochs} epochs...")
    start_time = datetime.now()
    
    try:
        history = classifier.train(
            train_generator, 
            val_generator, 
            epochs=epochs,
            model_save_path=model_save_path
        )
        
        end_time = datetime.now()
        training_duration = (end_time - start_time).total_seconds()
        
        print(f"\nTraining completed in {training_duration/60:.2f} minutes")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return False
    
    # Evaluate model
    print("\n7. Evaluating model...")
    try:
        metrics = classifier.evaluate_model(X_test, y_test)
        
        print("\nTest Set Results:")
        print(f"   Accuracy: {metrics['accuracy']:.4f}")
        print(f"   Precision: {metrics['precision']:.4f}")
        print(f"   Recall: {metrics['recall']:.4f}")
        print(f"   F1 Score: {metrics['f1_score']:.4f}")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return False
    
    # Save model and metadata
    print("\n8. Saving model and metadata...")
    try:
        metadata_path = model_save_path.replace('.h5', '_metadata.json')
        classifier.save_model(model_save_path, metadata_path)
        print(f"   Model saved to: {model_save_path}")
        print(f"   Metadata saved to: {metadata_path}")
        
    except Exception as e:
        print(f"Error saving model: {e}")
        return False
    
    # Plot training history
    print("\n9. Generating training plots...")
    try:
        history_path = 'training_history.png'
        classifier.plot_training_history(save_path=history_path)
        print(f"   Training history saved to: {history_path}")
        
    except Exception as e:
        print(f"Error generating plots: {e}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"Model: {model_save_path}")
    print(f"Training time: {training_duration/60:.2f} minutes")
    print(f"Final accuracy: {metrics['accuracy']:.4f}")
    print("=" * 60)
    
    return True

def main():
    """Main function to parse arguments and run training"""
    
    parser = argparse.ArgumentParser(description='Train ML Image Classifier Model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--model-type', type=str, default='custom', 
                       choices=['custom', 'vgg16', 'resnet50'], 
                       help='Type of model to train')
    parser.add_argument('--data-dir', type=str, default='data', 
                       help='Path to data directory')
    parser.add_argument('--model-path', type=str, default='models/best_model.h5', 
                       help='Path to save the trained model')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' does not exist")
        sys.exit(1)
    
    if args.epochs <= 0:
        print("Error: Number of epochs must be positive")
        sys.exit(1)
    
    # Run training
    success = train_model(
        epochs=args.epochs,
        model_type=args.model_type,
        data_dir=args.data_dir,
        model_save_path=args.model_path
    )
    
    if success:
        print("\nModel training completed successfully!")
        sys.exit(0)
    else:
        print("\nModel training failed!")
        sys.exit(1)

if __name__ == "__main__":
    main() 