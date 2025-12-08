"""
Auto-annotation Script for YOLO Format
Generates initial annotations using a pre-trained model that can be refined in labelImg.
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import argparse
from typing import List, Dict
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from preprocessing import CLAHEPreprocessor


class YOLOAutoAnnotator:
    """
    Auto-annotate images using a pre-trained YOLO model.
    Generates YOLO format txt files that can be refined in labelImg.
    """
    
    def __init__(
        self,
        model_path: str = 'yolov8s.pt',
        use_clahe: bool = False,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: int = 8,
        conf_threshold: float = 0.25,
        device: str = None,
        class_names: List[str] = None
    ):
        """
        Initialize the auto-annotator.
        
        Parameters:
        -----------
        model_path : str
            Path to YOLO model weights
        use_clahe : bool
            Whether to apply CLAHE preprocessing
        conf_threshold : float
            Confidence threshold for detections
        device : str
            Device to run inference on
        class_names : List[str]
            List of class names in order (for YOLO format)
        """
        print(f"Loading YOLO model: {model_path}")
        # Fix for PyTorch 2.6 compatibility
        import torch
        if hasattr(torch.serialization, 'add_safe_globals'):
            import torch.nn.modules.container
            torch.serialization.add_safe_globals([torch.nn.modules.container.Sequential])
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.use_clahe = use_clahe
        
        # Set device
        if device is None:
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
        
        # Set class names
        if class_names:
            self.class_names = class_names
        else:
            # Default to COCO classes
            self.class_names = list(self.model.names.values())
        
        print(f"Class names: {self.class_names}")
    
    def detect_and_convert(
        self,
        image: np.ndarray,
        filter_classes: List[int] = None
    ) -> List[Dict]:
        """
        Detect objects and convert to YOLO format.
        
        Parameters:
        -----------
        image : np.ndarray
            Input image (BGR format)
        filter_classes : List[int]
            List of class IDs to keep (None = keep all)
        
        Returns:
        --------
        List[Dict] : List of detections in YOLO format
            Each dict has: {class_id, x_center, y_center, width, height, confidence}
        """
        # Apply CLAHE if enabled
        if self.use_clahe and self.preprocessor:
            image = self.preprocessor.process(image)
        
        # Get image dimensions
        img_height, img_width = image.shape[:2]
        
        # Run inference
        results = self.model(
            image,
            conf=self.conf_threshold,
            device=self.device,
            verbose=False
        )[0]
        
        # Extract detections and convert to YOLO format
        yolo_annotations = []
        boxes = results.boxes
        
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i])
            
            # Filter classes if requested
            if filter_classes is not None and class_id not in filter_classes:
                continue
            
            # Get bbox in xyxy format [x1, y1, x2, y2]
            bbox = boxes.xyxy[i].cpu().numpy()
            confidence = float(boxes.conf[i])
            
            # Convert to YOLO format (normalized x_center, y_center, width, height)
            x1, y1, x2, y2 = bbox
            x_center = ((x1 + x2) / 2) / img_width
            y_center = ((y1 + y2) / 2) / img_height
            width = (x2 - x1) / img_width
            height = (y2 - y1) / img_height
            
            yolo_annotations.append({
                'class_id': class_id,
                'x_center': x_center,
                'y_center': y_center,
                'width': width,
                'height': height,
                'confidence': confidence
            })
        
        return yolo_annotations
    
    def save_yolo_annotation(
        self,
        annotations: List[Dict],
        output_path: str
    ):
        """
        Save annotations in YOLO format (.txt file).
        
        Format: class_id x_center y_center width height (one per line)
        All values except class_id are normalized to [0, 1]
        """
        with open(output_path, 'w') as f:
            for ann in annotations:
                line = f"{ann['class_id']} {ann['x_center']:.6f} {ann['y_center']:.6f} {ann['width']:.6f} {ann['height']:.6f}\n"
                f.write(line)
    
    def annotate_image(
        self,
        image_path: str,
        output_dir: str,
        filter_classes: List[int] = None,
        save_visualization: bool = False
    ) -> int:
        """
        Auto-annotate a single image.
        
        Returns:
        --------
        int : Number of annotations created
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"ERROR: Could not read image: {image_path}")
            return 0
        
        # Detect and convert
        annotations = self.detect_and_convert(image, filter_classes)
        
        # Save annotation file
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_name = Path(image_path).stem
        annotation_path = output_dir / f"{image_name}.txt"
        self.save_yolo_annotation(annotations, annotation_path)
        
        print(f"✓ {image_name}: {len(annotations)} annotations saved to {annotation_path}")
        
        # Save visualization if requested
        if save_visualization:
            results = self.model(image, conf=self.conf_threshold, device=self.device, verbose=False)[0]
            annotated_img = results.plot()
            viz_path = output_dir / f"{image_name}_viz.jpg"
            cv2.imwrite(str(viz_path), annotated_img)
        
        return len(annotations)
    
    def annotate_directory(
        self,
        input_dir: str,
        output_dir: str,
        filter_classes: List[int] = None,
        save_visualization: bool = False,
        image_extensions: List[str] = None
    ):
        """
        Auto-annotate all images in a directory.
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"ERROR: Input directory does not exist: {input_dir}")
            return
        
        # Find all images
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        image_files = sorted(set(image_files))
        
        if not image_files:
            print(f"ERROR: No images found in {input_dir}")
            return
        
        print(f"\nFound {len(image_files)} images")
        print(f"Output directory: {output_dir}")
        print(f"Confidence threshold: {self.conf_threshold}")
        if filter_classes:
            print(f"Filtering classes: {filter_classes}")
        print("\nProcessing...\n")
        
        # Process each image
        total_annotations = 0
        for i, image_file in enumerate(image_files, 1):
            print(f"[{i}/{len(image_files)}] ", end="")
            num_annotations = self.annotate_image(
                str(image_file),
                output_dir,
                filter_classes,
                save_visualization
            )
            total_annotations += num_annotations
        
        print(f"\n✓ Done! Processed {len(image_files)} images")
        print(f"✓ Total annotations: {total_annotations}")
        print(f"✓ Average per image: {total_annotations/len(image_files):.1f}")
        print(f"\nYou can now open these annotations in labelImg for refinement:")
        print(f"  labelImg {input_dir} data/classes.txt {output_dir}")


def main():
    """Command-line interface for auto-annotation."""
    parser = argparse.ArgumentParser(
        description='Auto-annotate images using pre-trained YOLO model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-annotate all images in a directory (person class only)
  python auto_annotate.py data/to_annotate/ -o data/labels/ --person-only
  
  # Use CLAHE preprocessing for low-light images
  python auto_annotate.py data/to_annotate/ -o data/labels/ --clahe --person-only
  
  # Lower confidence threshold to get more detections
  python auto_annotate.py data/to_annotate/ -o data/labels/ --conf 0.15 --person-only
  
  # Save visualization images
  python auto_annotate.py data/to_annotate/ -o data/labels/ --viz --person-only
        """
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory containing images to annotate'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='Directory to save YOLO format annotations'
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
        help='Enable CLAHE preprocessing for low-light images'
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
        '--person-only',
        action='store_true',
        help='Only annotate person class (COCO class_id=0)'
    )
    parser.add_argument(
        '--classes',
        type=str,
        help='Comma-separated list of class IDs to include (e.g., "0,1,2")'
    )
    parser.add_argument(
        '--viz',
        action='store_true',
        help='Save visualization images'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'mps', 'cuda'],
        help='Device for inference (default: auto-detect)'
    )
    
    args = parser.parse_args()
    
    # Parse filter classes
    filter_classes = None
    if args.person_only:
        filter_classes = [0]  # COCO person class
    elif args.classes:
        filter_classes = [int(c.strip()) for c in args.classes.split(',')]
    
    # Create annotator
    annotator = YOLOAutoAnnotator(
        model_path=args.model,
        use_clahe=args.clahe,
        clahe_clip_limit=args.clip_limit,
        clahe_tile_size=args.tile_size,
        conf_threshold=args.conf,
        device=args.device
    )
    
    # Run auto-annotation
    annotator.annotate_directory(
        args.input_dir,
        args.output_dir,
        filter_classes=filter_classes,
        save_visualization=args.viz
    )
    
    return 0


if __name__ == '__main__':
    exit(main())
