#!/usr/bin/env python3
"""
Quick test script to verify the setup works correctly.
This will download YOLOv8s model and test basic functionality.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

# Add scripts to path
sys.path.append('scripts')

from preprocessing import CLAHEPreprocessor
from ultralytics import YOLO
import torch


def create_test_image():
    """Create a simple test image with a person-like shape."""
    # Create a dark image (low-light simulation)
    img = np.ones((480, 640, 3), dtype=np.uint8) * 30  # Dark background
    
    # Draw a simple person-like shape (rectangle for body, circle for head)
    # Make it slightly brighter than background
    cv2.rectangle(img, (250, 200), (390, 400), (60, 60, 60), -1)  # Body
    cv2.circle(img, (320, 150), 40, (60, 60, 60), -1)  # Head
    
    # Add some noise to simulate low-light conditions
    noise = np.random.normal(0, 10, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)
    
    return img


def test_clahe():
    """Test CLAHE preprocessing."""
    print("=" * 60)
    print("TEST 1: CLAHE Preprocessing")
    print("=" * 60)
    
    try:
        preprocessor = CLAHEPreprocessor(clip_limit=2.0, tile_grid_size=(8, 8))
        test_img = create_test_image()
        enhanced = preprocessor.process(test_img)
        
        # Save test images
        output_dir = Path('results/test')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cv2.imwrite(str(output_dir / 'test_original.jpg'), test_img)
        cv2.imwrite(str(output_dir / 'test_enhanced.jpg'), enhanced)
        
        print("✓ CLAHE preprocessing works!")
        print(f"  Original image brightness: {test_img.mean():.1f}")
        print(f"  Enhanced image brightness: {enhanced.mean():.1f}")
        print(f"  Saved to: {output_dir}")
        return True
    except Exception as e:
        print(f"✗ CLAHE test failed: {e}")
        return False


def test_yolo():
    """Test YOLO model loading."""
    print("\n" + "=" * 60)
    print("TEST 2: YOLOv8 Model Loading")
    print("=" * 60)
    
    try:
        print("Downloading YOLOv8s model (this may take a minute)...")
        model = YOLO('yolov8s.pt')
        
        # Check device
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
        
        print(f"✓ YOLOv8s model loaded successfully!")
        print(f"  Device: {device} (Apple Silicon GPU)" if device == 'mps' else f"  Device: {device}")
        print(f"  Model: yolov8s.pt")
        return True
    except Exception as e:
        print(f"✗ YOLO test failed: {e}")
        return False


def test_detection():
    """Test full detection pipeline."""
    print("\n" + "=" * 60)
    print("TEST 3: Full Detection Pipeline")
    print("=" * 60)
    
    try:
        # Create test image
        test_img = create_test_image()
        
        # Try detection with CLAHE
        from detect import PeopleDetector
        
        print("Testing detection with CLAHE preprocessing...")
        detector = PeopleDetector(
            model_path='yolov8s.pt',
            use_clahe=True,
            conf_threshold=0.25
        )
        
        annotated, detections, inf_time = detector.detect(test_img)
        
        # Save result
        output_dir = Path('results/test')
        output_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_dir / 'test_detection.jpg'), annotated)
        
        print(f"✓ Detection pipeline works!")
        print(f"  Detections found: {len(detections)}")
        print(f"  Inference time: {inf_time:.3f}s")
        print(f"  Result saved to: {output_dir / 'test_detection.jpg'}")
        
        if len(detections) > 0:
            print(f"  Note: Simple test shape detected (shows system works)")
        else:
            print(f"  Note: No detections (expected for simple test shape)")
        
        return True
    except Exception as e:
        print(f"✗ Detection test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("LOW-LIGHT YOLO SETUP VERIFICATION")
    print("="*60)
    print(f"Python version: {sys.version.split()[0]}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print()
    
    results = []
    
    # Run tests
    results.append(("CLAHE Preprocessing", test_clahe()))
    results.append(("YOLOv8 Model Loading", test_yolo()))
    results.append(("Full Detection Pipeline", test_detection()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name:<30} {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✅ All tests passed! Setup is ready to use.")
        print("\nNext steps:")
        print("1. Add your own low-light images to data/images/")
        print("2. Run: python scripts/detect.py data/images/your_image.jpg --compare")
        print("3. Check results/ directory for output")
    else:
        print("\n⚠️  Some tests failed. Please check the error messages above.")
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
