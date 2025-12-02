"""
People Detection Script with Optional CLAHE Preprocessing
Tests pretrained YOLOv8 with and without low-light enhancement.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
import time
from typing import List, Tuple
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from preprocessing import CLAHEPreprocessor


class PeopleDetector:
    """
    People detection using YOLOv8 with optional CLAHE preprocessing.
    """
    
    def __init__(
        self,
        model_path: str = 'yolov8s.pt',
        use_clahe: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: int = 8,
        conf_threshold: float = 0.25,
        device: str = None
    ):
        """
        Initialize the people detector.
        
        Parameters:
        -----------
        model_path : str
            Path to YOLO model weights (default: yolov8s.pt - downloads if not exists)
        use_clahe : bool
            Whether to apply CLAHE preprocessing
        clahe_clip_limit : float
            CLAHE clip limit parameter
        clahe_tile_size : int
            CLAHE tile grid size
        conf_threshold : float
            Confidence threshold for detections (0.0-1.0)
        device : str
            Device to run inference on ('cpu', 'mps', 'cuda', or None for auto)
        """
        print(f"Loading YOLO model: {model_path}")
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.use_clahe = use_clahe
        
        # Set device
        if device is None:
            # Auto-detect: prefer MPS on Mac, then CUDA, then CPU
            import torch
            if torch.backends.mps.is_available():
                self.device = 'mps'
            elif torch.cuda.is_available():
                self.device = 'cuda'
            else:
                self.device = 'cpu'
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        
        # Initialize CLAHE preprocessor if needed
        if use_clahe:
            self.preprocessor = CLAHEPreprocessor(
                clip_limit=clahe_clip_limit,
                tile_grid_size=(clahe_tile_size, clahe_tile_size)
            )
            print(f"CLAHE enabled (clip_limit={clahe_clip_limit}, tile_size={clahe_tile_size})")
        else:
            self.preprocessor = None
            print("CLAHE disabled")
    
    def detect(
        self,
        image: np.ndarray,
        filter_person_class: bool = True
    ) -> Tuple[np.ndarray, List[dict], float]:
        """
        Detect people in an image.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image (BGR format)
        filter_person_class : bool
            If True, returns only 'person' detections (class_id=0 in COCO)
        
        Returns:
        --------
        tuple : (annotated_image, detections, inference_time)
            - annotated_image: Image with bounding boxes drawn
            - detections: List of detection dicts with keys:
                          {bbox, confidence, class_id, class_name}
            - inference_time: Time taken for inference in seconds
        """
        # Apply CLAHE if enabled
        if self.use_clahe and self.preprocessor:
            image = self.preprocessor.process(image)
        
        # Run inference
        start_time = time.time()
        results = self.model(
            image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )[0]
        inference_time = time.time() - start_time
        
        # Extract detections
        detections = []
        boxes = results.boxes
        
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i])
            
            # Filter for person class if requested (class_id=0 in COCO)
            if filter_person_class and class_id != 0:
                continue
            
            bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
            confidence = float(boxes.conf[i])
            class_name = results.names[class_id]
            
            detections.append({
                'bbox': bbox.tolist(),
                'confidence': confidence,
                'class_id': class_id,
                'class_name': class_name
            })
        
        # Get annotated image
        annotated_image = results.plot()
        
        return annotated_image, detections, inference_time
    
    def detect_file(
        self,
        input_path: str,
        output_path: str = None,
        show_result: bool = False
    ) -> Tuple[int, float]:
        """
        Detect people in an image file.
        
        Returns:
        --------
        tuple : (num_detections, inference_time)
        """
        # Read image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not read image: {input_path}")
        
        # Detect
        annotated, detections, inf_time = self.detect(image)
        
        # Print results
        print(f"\nImage: {Path(input_path).name}")
        print(f"Detections: {len(detections)} people")
        print(f"Inference time: {inf_time:.3f}s")
        for i, det in enumerate(detections, 1):
            print(f"  {i}. {det['class_name']}: {det['confidence']:.2f}")
        
        # Save if output path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), annotated)
            print(f"Saved result to: {output_path}")
        
        # Show if requested
        if show_result:
            cv2.imshow('Detection Result', annotated)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return len(detections), inf_time
    
    def compare_with_without_clahe(
        self,
        input_path: str,
        output_dir: str = None
    ):
        """
        Compare detection results with and without CLAHE.
        Useful for evaluating CLAHE effectiveness.
        """
        print("\n" + "="*60)
        print("Comparing Detection: With vs Without CLAHE")
        print("="*60)
        
        # Read image
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Could not read image: {input_path}")
        
        # Detect without CLAHE
        original_use_clahe = self.use_clahe
        self.use_clahe = False
        annotated_no_clahe, det_no_clahe, time_no_clahe = self.detect(image)
        
        # Detect with CLAHE
        self.use_clahe = True
        if self.preprocessor is None:
            self.preprocessor = CLAHEPreprocessor()
        annotated_clahe, det_clahe, time_clahe = self.detect(image)
        
        # Restore original setting
        self.use_clahe = original_use_clahe
        
        # Print comparison
        print(f"\nWithout CLAHE:")
        print(f"  Detections: {len(det_no_clahe)}")
        print(f"  Inference time: {time_no_clahe:.3f}s")
        
        print(f"\nWith CLAHE:")
        print(f"  Detections: {len(det_clahe)}")
        print(f"  Inference time: {time_clahe:.3f}s")
        
        print(f"\nDifference: {len(det_clahe) - len(det_no_clahe):+d} detections")
        
        # Save comparison if output directory provided
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            filename = Path(input_path).stem
            cv2.imwrite(str(output_dir / f"{filename}_no_clahe.jpg"), annotated_no_clahe)
            cv2.imwrite(str(output_dir / f"{filename}_with_clahe.jpg"), annotated_clahe)
            print(f"\nSaved comparison images to: {output_dir}")


def main():
    """Command-line interface for people detection."""
    parser = argparse.ArgumentParser(
        description='People detection with YOLOv8 and optional CLAHE preprocessing'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Input image file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output image file'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='yolov8s.pt',
        help='YOLO model path (default: yolov8s.pt)'
    )
    parser.add_argument(
        '--clahe',
        action='store_true',
        help='Enable CLAHE preprocessing'
    )
    parser.add_argument(
        '--clip-limit',
        type=float,
        default=2.0,
        help='CLAHE clip limit (default: 2.0)'
    )
    parser.add_argument(
        '--tile-size',
        type=int,
        default=8,
        help='CLAHE tile grid size (default: 8)'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare results with and without CLAHE'
    )
    parser.add_argument(
        '--show',
        action='store_true',
        help='Display result image'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'mps', 'cuda'],
        help='Device for inference (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Create detector
    detector = PeopleDetector(
        model_path=args.model,
        use_clahe=args.clahe,
        clahe_clip_limit=args.clip_limit,
        clahe_tile_size=args.tile_size,
        conf_threshold=args.conf,
        device=args.device
    )
    
    # Run detection
    if args.compare:
        detector.compare_with_without_clahe(
            args.input,
            output_dir=args.output if args.output else 'results/comparison'
        )
    else:
        detector.detect_file(
            args.input,
            output_path=args.output,
            show_result=args.show
        )
    
    return 0


if __name__ == '__main__':
    exit(main())
