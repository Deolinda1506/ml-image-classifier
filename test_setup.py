#!/usr/bin/env python3
"""
Test script to verify the ML pipeline setup
"""

import os
import sys
import importlib
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    required_modules = [
        'tensorflow',
        'numpy',
        'pandas',
        'sklearn',
        'cv2',
        'PIL',
        'matplotlib',
        'seaborn',
        'flask',
        'psutil'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            importlib.import_module(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\nFailed imports: {failed_imports}")
        return False
    
    print("All imports successful!")
    return True

def test_custom_modules():
    """Test if custom modules can be imported"""
    print("\nTesting custom modules...")
    
    # Add src to path
    sys.path.append('src')
    
    try:
        from preprocessing import ImagePreprocessor
        print("  ✓ ImagePreprocessor")
    except ImportError as e:
        print(f"  ✗ ImagePreprocessor: {e}")
        return False
    
    try:
        from model import ImageClassifier
        print("  ✓ ImageClassifier")
    except ImportError as e:
        print(f"  ✗ ImageClassifier: {e}")
        return False
    
    try:
        from prediction import PredictionService
        print("  ✓ PredictionService")
    except ImportError as e:
        print(f"  ✗ PredictionService: {e}")
        return False
    
    print("All custom modules imported successfully!")
    return True

def test_data_structure():
    """Test if data directory structure is correct"""
    print("\nTesting data structure...")
    
    required_dirs = ['data/train', 'data/test']
    required_files = ['app.py', 'requirements.txt', 'README.md']
    
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}")
        else:
            print(f"  ✗ {dir_path} (missing)")
            return False
    
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (missing)")
            return False
    
    # Check if there are images in data directories
    train_images = len([f for f in os.listdir('data/train') if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    test_images = len([f for f in os.listdir('data/test') if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    
    print(f"  ✓ Training images: {train_images}")
    print(f"  ✓ Test images: {test_images}")
    
    if train_images == 0 or test_images == 0:
        print("  ⚠ Warning: No images found in data directories")
    
    return True

def test_model_creation():
    """Test if model can be created"""
    print("\nTesting model creation...")
    
    try:
        sys.path.append('src')
        from model import ImageClassifier
        
        classifier = ImageClassifier(model_type='custom')
        model = classifier.build_model()
        
        print(f"  ✓ Model created successfully")
        print(f"  ✓ Model type: {classifier.model_type}")
        print(f"  ✓ Input shape: {model.input_shape}")
        print(f"  ✓ Output shape: {model.output_shape}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        return False

def test_preprocessing():
    """Test if preprocessing works"""
    print("\nTesting preprocessing...")
    
    try:
        sys.path.append('src')
        from preprocessing import ImagePreprocessor
        
        preprocessor = ImagePreprocessor()
        
        # Test with a sample image if available
        train_dir = 'data/train'
        if os.path.exists(train_dir):
            images = [f for f in os.listdir(train_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                sample_image = os.path.join(train_dir, images[0])
                processed = preprocessor.load_and_preprocess_image(sample_image)
                print(f"  ✓ Image preprocessing successful")
                print(f"  ✓ Processed image shape: {processed.shape}")
                return True
        
        print("  ✓ Preprocessor initialized (no images to test)")
        return True
        
    except Exception as e:
        print(f"  ✗ Preprocessing failed: {e}")
        return False

def test_flask_app():
    """Test if Flask app can be imported"""
    print("\nTesting Flask app...")
    
    try:
        # This is a basic test - we don't actually run the app
        import app
        print("  ✓ Flask app can be imported")
        return True
        
    except Exception as e:
        print(f"  ✗ Flask app import failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("ML PIPELINE SETUP TEST")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_custom_modules,
        test_data_structure,
        test_model_creation,
        test_preprocessing,
        test_flask_app
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 60)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    print("=" * 60)
    
    if passed == total:
        print("🎉 All tests passed! Your ML pipeline is ready to use.")
        print("\nNext steps:")
        print("1. Train the model: python train_model.py")
        print("2. Start the application: python app.py")
        print("3. Open http://localhost:5000 in your browser")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\nCommon solutions:")
        print("1. Install missing dependencies: pip install -r requirements.txt")
        print("2. Ensure data directory structure is correct")
        print("3. Check Python version (3.8+ required)")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 