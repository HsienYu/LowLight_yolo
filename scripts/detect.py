"""
People Detection Script with Optional CLAHE & Zero-DCE++ Preprocessing
Tests pretrained YOLOv8 with low-light enhancement methods.

Enhancement options:
- CLAHE (classical method)
- Zero-DCE++ (deep learning method)
- Hybrid (Zero-DCE++ + CLAHE)
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO, RTDETR
import argparse
import time
from typing import List, Tuple
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
from preprocessing import CLAHEPreprocessor
try:
    from zero_dce import ZeroDCEEnhancer
    from yola import YOLAEnhancer
    from hybrid_detector import SequentialDetector, AdaptiveDetector, EnsembleDetector
    ZERO_DCE_AVAILABLE = True
    YOLA_AVAILABLE = True
    HYBRID_DETECTORS_AVAILABLE = True
except ImportError:
    ZERO_DCE_AVAILABLE = False
    YOLA_AVAILABLE = False
    HYBRID_DETECTORS_AVAILABLE = False
    print("WARNING: Advanced enhancement modules not available. Install requirements.")

# ZED SDK support
try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    ZED_AVAILABLE = False


class PeopleDetector:
    """
    People detection using multiple model architectures with optional enhancement preprocessing.

    Supported models:
    - YOLOv8 (yolov8s.pt, yolov8m.pt, etc.)
    - YOLOv10 (yolov10s.pt, yolov10m.pt, etc.)
    - RT-DETR (rtdetr-l.pt, rtdetr-x.pt)

    Supported enhancements:
    - CLAHE (classical)
    - Zero-DCE++ (deep learning)
    - YOLA (NeurIPS 2024 deep learning)
    - Hybrid methods

    Preset Configurations:
    - max_accuracy: RT-DETR-X + CLAHE (10 detections, 0.79 confidence, 1.186s)
    - balanced: YOLOv10m + CLAHE (3 detections, 0.66 confidence, 0.919s)
    - real_time: YOLOv8m + CLAHE (4 detections, 0.75 confidence, 0.626s)
    - yola_max: RT-DETR-X + YOLA (High-precision low-light detection)
    """

    # Preset configurations based on theatre testing results
    PRESETS = {
        # Basic CLAHE presets
        'max_accuracy': {
            'model_path': 'models/rtdetr-x.pt',
            'model_type': 'rtdetr',
            'use_clahe': True,
            'use_zero_dce': False,
            'use_yola': False,
            'hybrid_mode': None,
            'description': 'Maximum accuracy: RT-DETR-X + CLAHE (10 people, 0.79 conf, 1.186s)'
        },
        'balanced': {
            'model_path': 'models/yolov10m.pt',
            'model_type': 'yolo',
            'use_clahe': True,
            'use_zero_dce': False,
            'use_yola': False,
            'hybrid_mode': None,
            'description': 'Balanced performance: YOLOv10m + CLAHE (3 people, 0.66 conf, 0.919s)'
        },
        'real_time': {
            'model_path': 'models/yolov8m.pt',
            'model_type': 'yolo',
            'use_clahe': True,
            'use_zero_dce': False,
            'use_yola': False,
            'hybrid_mode': None,
            'description': 'Real-time speed: YOLOv8m + CLAHE (4 people, 0.75 conf, 0.626s)'
        },
        # Advanced Zero-DCE++ presets
        'ultra_accuracy': {
            'model_path': 'models/rtdetr-x.pt',
            'model_type': 'rtdetr',
            'use_clahe': False,
            'use_zero_dce': True,
            'use_yola': False,
            'hybrid_mode': 'sequential',
            'description': 'Ultra accuracy: RT-DETR-X + Zero-DCE++ Sequential (best quality, slower)'
        },
        'adaptive_smart': {
            'model_path': 'models/yolov8m.pt',
            'model_type': 'yolo',
            'use_clahe': False,
            'use_zero_dce': True,
            'use_yola': False,
            'hybrid_mode': 'adaptive',
            'description': 'Smart adaptive: YOLOv8m + Adaptive enhancement (auto-selects method)'
        },
        'ensemble_max': {
            'model_path': 'models/yolov8m.pt',
            'model_type': 'yolo',
            'use_clahe': False,
            'use_zero_dce': True,
            'use_yola': False,
            'hybrid_mode': 'ensemble',
            'description': 'Ensemble maximum: YOLOv8m + Multi-enhancement fusion (highest accuracy)'
        },
        # YOLA Presets
        'yola_max': {
            'model_path': 'models/rtdetr-x.pt',
            'model_type': 'rtdetr',
            'use_clahe': False,
            'use_zero_dce': False,
            'use_yola': True,
            'hybrid_mode': None,
            'description': 'YOLA Max: RT-DETR-X + YOLA Enhancement (NeurIPS 2024)'
        },
        'yola_balanced': {
            'model_path': 'models/yolov10m.pt',
            'model_type': 'yolo',
            'use_clahe': False,
            'use_zero_dce': False,
            'use_yola': True,
            'hybrid_mode': None,
            'description': 'YOLA Balanced: YOLOv10m + YOLA Enhancement (NeurIPS 2024)'
        }
    }

    def __init__(
        self,
        model_path: str = 'models/yolov8s.pt',
        model_type: str = 'auto',
        enhancement: str = 'none',
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: int = 8,
        zero_dce_weights: str = None,
        yola_weights: str = None,
        conf_threshold: float = 0.25,
        device: str = None,
        # Legacy parameter for backward compatibility
        use_clahe: bool = None,
        # Preset configuration
        preset: str = None,
        # Image mirroring
        mirror: bool = True
    ):
        """
        Initialize the people detector.

        Parameters:
        -----------
        model_path : str
            Path to model weights (e.g., models/yolov8s.pt, models/rtdetr-l.pt, models/yolov10s.pt)
        model_type : str
            Model architecture type ('yolo', 'rtdetr', or 'auto' for auto-detection)
        enhancement : str
            Enhancement method ('none', 'clahe', 'zero_dce', 'yola', 'hybrid')
        use_clahe : bool
            Legacy parameter - whether to apply CLAHE preprocessing
        clahe_clip_limit : float
            CLAHE clip limit parameter
        clahe_tile_size : int
            CLAHE tile grid size
        conf_threshold : float
            Confidence threshold for detections (0.0-1.0)
        device : str
            Device to run inference on ('cpu', 'mps', 'cuda', or None for auto)
        preset : str
            Preset configuration ('max_accuracy', 'balanced', 'real_time', or None)
        mirror : bool
            Whether to horizontally flip the input image (for natural mirror view)
        """

        # Apply preset configuration if specified
        if preset and preset in self.PRESETS:
            preset_config = self.PRESETS[preset]
            model_path = preset_config['model_path']
            model_type = preset_config['model_type']
            use_clahe = preset_config['use_clahe']
            use_zero_dce = preset_config.get('use_zero_dce', False)
            use_yola = preset_config.get('use_yola', False)
            hybrid_mode = preset_config.get('hybrid_mode', None)
            print(f"Using preset '{preset}': {preset_config['description']}")

            # Check availability of advanced features
            if (use_zero_dce or hybrid_mode) and not HYBRID_DETECTORS_AVAILABLE:
                print("WARNING: Advanced features require Zero-DCE++ and hybrid detectors. Falling back to CLAHE.")
                use_clahe = True
                use_zero_dce = False
                hybrid_mode = None
            
            if use_yola and not YOLA_AVAILABLE:
                print("WARNING: YOLA enhancement not available. Falling back to CLAHE.")
                use_clahe = True
                use_yola = False

        else:
            use_zero_dce = False
            use_yola = False
            hybrid_mode = None

        # Auto-detect model type from filename if not specified
        if model_type == 'auto':
            model_path_lower = model_path.lower()
            if 'rtdetr' in model_path_lower:
                model_type = 'rtdetr'
            elif 'yolo' in model_path_lower or model_path_lower.endswith('.pt'):
                model_type = 'yolo'
            else:
                model_type = 'yolo'  # Default fallback

        self.model_type = model_type
        self.use_zero_dce = use_zero_dce
        self.use_yola = use_yola
        self.hybrid_mode = hybrid_mode

        # Initialize hybrid detector if specified
        if hybrid_mode and HYBRID_DETECTORS_AVAILABLE:
            print(f"Loading {hybrid_mode.upper()} hybrid detector with {model_type.upper()}")
            # Use default Zero-DCE weights path if not specified
            zdce_weights = zero_dce_weights if zero_dce_weights else 'models/zero_dce_plus.pth'
            if hybrid_mode == 'sequential':
                self.hybrid_detector = SequentialDetector(
                    yolo_model=model_path,
                    zero_dce_weights=zdce_weights,
                    device=device if device else 'auto'
                )
            elif hybrid_mode == 'adaptive':
                self.hybrid_detector = AdaptiveDetector(
                    yolo_model=model_path,
                    zero_dce_weights=zdce_weights,
                    device=device if device else 'auto'
                )
            elif hybrid_mode == 'ensemble':
                self.hybrid_detector = EnsembleDetector(
                    yolo_model=model_path,
                    zero_dce_weights=zdce_weights,
                    device=device if device else 'auto'
                )
            self.model = None  # Will use hybrid detector instead
        else:
            # Load standard model
            print(f"Loading {model_type.upper()} model: {model_path}")
            if model_type == 'rtdetr':
                self.model = RTDETR(model_path)
            else:  # yolo (covers YOLOv8, YOLOv10, etc.)
                self.model = YOLO(model_path)
            self.hybrid_detector = None

        self.conf_threshold = conf_threshold
        self.use_clahe = use_clahe
        self.mirror = mirror
        
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
        
        # Initialize YOLA enhancer if needed
        self.yola_weights = yola_weights
        if use_yola and YOLA_AVAILABLE:
             # Default to models/yola.pth if not specified
             if not self.yola_weights:
                 default_path = Path('models/yola.pth')
                 if default_path.exists():
                     self.yola_weights = str(default_path)
             
             self.yola_enhancer = YOLAEnhancer(model_path=self.yola_weights, device=self.device)
             print(f"YOLA enhancement enabled (weights: {self.yola_weights if self.yola_weights else 'Random - WARNING: Place weights in models/yola.pth for best results'})")

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
        
        # Mirror setting
        if mirror:
            print("Image mirroring enabled (horizontal flip for natural mirror view)")
    
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
        start_time = time.time()

        # Apply mirroring first (before any processing) for natural mirror view
        if self.mirror:
            image = cv2.flip(image, 1)  # Horizontal flip

        # Use hybrid detector if available
        if self.hybrid_detector:
            if self.hybrid_mode == 'adaptive':
                # AdaptiveDetector returns (results, enhanced, strategy)
                results, enhanced_image, strategy = self.hybrid_detector.detect(
                    image,
                    conf=self.conf_threshold
                )
            elif self.hybrid_mode == 'ensemble':
                # EnsembleDetector returns (results, enhanced_images_dict)
                results, enhanced_images_dict = self.hybrid_detector.detect(
                    image,
                    conf=self.conf_threshold
                )
                # Extract the best enhanced image from dictionary (use combined for best quality)
                enhanced_image = enhanced_images_dict.get('combined', enhanced_images_dict.get('zero_dce', image))
            else:
                # Sequential return (results, enhanced)
                results, enhanced_image = self.hybrid_detector.detect(
                    image,
                    conf=self.conf_threshold
                )
            # Convert hybrid detector results to standard format
            results = results[0] if isinstance(results, list) else results
            processed_image = enhanced_image  # Use enhanced image for drawing
            inference_time = time.time() - start_time
        else:
            # Standard detection pipeline
            processed_image = image

            # Apply CLAHE if enabled
            if self.use_clahe and self.preprocessor:
                processed_image = self.preprocessor.process(processed_image)

            # Apply Zero-DCE++ if enabled and available
            if self.use_zero_dce and ZERO_DCE_AVAILABLE:
                if not hasattr(self, 'zero_dce_enhancer'):
                    self.zero_dce_enhancer = ZeroDCEEnhancer(device=self.device)
                processed_image = self.zero_dce_enhancer.enhance(processed_image)

            # Apply YOLA if enabled and available
            if self.use_yola and YOLA_AVAILABLE and hasattr(self, 'yola_enhancer'):
                processed_image = self.yola_enhancer.enhance(processed_image)

            # Run inference
            results = self.model(
                processed_image,
                conf=self.conf_threshold,
                device=self.device,
                verbose=False
            )[0]
            inference_time = time.time() - start_time
        
        # Extract detections
        detections = []
        boxes = results.boxes

        # Debug: Print class mapping for first run
        if hasattr(self, '_debug_printed') is False:
            print(f"\nDebug - {self.model_type.upper()} Class Mapping:")
            for class_id, class_name in results.names.items():
                print(f"  Class {class_id}: {class_name}")
            self._debug_printed = True

        # Track indices of person detections for filtered visualization
        person_indices = []
        
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i])
            class_name = results.names[class_id]

            # Filter for person class if requested
            # Use class name matching for universal compatibility across model types
            if filter_person_class and class_name.lower() != 'person':
                continue

            person_indices.append(i)
            bbox = boxes.xyxy[i].cpu().numpy()  # [x1, y1, x2, y2]
            confidence = float(boxes.conf[i])

            # Calculate center bottom point (feet position)
            x1, y1, x2, y2 = bbox
            center_bottom_x = int((x1 + x2) / 2)
            center_bottom_y = int(y2)

            detections.append({
                'bbox': bbox.tolist(),
                'center_bottom': (center_bottom_x, center_bottom_y),
                'confidence': confidence,
                'class_id': class_id,
                'class_name': class_name
            })
        
        # Get annotated image - filter visualization to show only person detections
        if filter_person_class:
            # Always use custom drawing for consistent green thin boxes when filtering for persons
            annotated_image = processed_image.copy()
            for i in person_indices:
                bbox = boxes.xyxy[i].cpu().numpy()
                confidence = float(boxes.conf[i])
                class_name = results.names[int(boxes.cls[i])]
                
                # Draw thin green bounding box (thickness=1 for thin line)
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Calculate and draw center bottom point
                center_bottom_x = (x1 + x2) // 2
                center_bottom_y = y2
                cv2.circle(annotated_image, (center_bottom_x, center_bottom_y), 5, (0, 0, 255), -1)  # Red filled circle
                
                # Draw label background and text (include center_bottom coords)
                label = f"{class_name} {confidence:.2f} ({center_bottom_x},{center_bottom_y})"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated_image, (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0], y1), (0, 255, 0),  -1)
                cv2.putText(annotated_image, label, (x1, y1 - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:
            # Use default plotting if not filtering
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
            cb = det['center_bottom']
            print(f"  {i}. {det['class_name']}: {det['confidence']:.2f} | center_bottom: ({cb[0]}, {cb[1]})")
        
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
    
    def detect_video(
        self,
        video_path: str,
        output_path: str = None,
        show_window: bool = True,
        save_stats: bool = False
    ):
        """
        Detect people in a video file.
        
        Parameters:
        -----------
        video_path : str
            Path to input video file
        output_path : str
            Path to save output video (optional)
        show_window : bool
            Display detection window in real-time
        save_stats : bool
            Save detection statistics to CSV
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nVideo: {Path(video_path).name}")
        print(f"Resolution: {width}x{height} @ {fps}fps")
        print(f"Total frames: {total_frames}")
        print(f"Press 'q' to quit, 'p' to pause/resume\n")
        
        # Setup video writer if output specified
        out = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Statistics
        frame_count = 0
        total_detections = 0
        total_time = 0
        stats = []
        paused = False
        
        try:
            from tqdm import tqdm
            pbar = tqdm(total=total_frames, desc="Processing")
        except ImportError:
            pbar = None
        
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect
                annotated, detections, inf_time = self.detect(frame)
                
                # Update statistics
                frame_count += 1
                num_detections = len(detections)
                total_detections += num_detections
                total_time += inf_time
                
                # Add frame info overlay
                cv2.putText(annotated, f"Frame: {frame_count}/{total_frames}", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"Detections: {num_detections}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated, f"FPS: {1/inf_time:.1f}", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Save stats
                if save_stats:
                    stats.append({
                        'frame': frame_count,
                        'detections': num_detections,
                        'inference_time': inf_time
                    })
                
                # Write to output video
                if out:
                    out.write(annotated)
                
                if pbar:
                    pbar.update(1)
            
            # Show window
            if show_window:
                cv2.imshow('People Detection', annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"{'Paused' if paused else 'Resumed'}")
        
        # Cleanup
        cap.release()
        if out:
            out.release()
        if show_window:
            cv2.destroyAllWindows()
        if pbar:
            pbar.close()
        
        # Print summary
        print(f"\n" + "="*60)
        print("VIDEO DETECTION SUMMARY")
        print("="*60)
        print(f"Processed frames: {frame_count}/{total_frames}")
        print(f"Total detections: {total_detections}")
        print(f"Average detections/frame: {total_detections/frame_count:.2f}")
        print(f"Average FPS: {frame_count/total_time:.1f}")
        
        if output_path and out:
            print(f"\nOutput video saved to: {output_path}")
        
        # Save statistics
        if save_stats and stats:
            import csv
            stats_path = Path(video_path).parent / f"{Path(video_path).stem}_stats.csv"
            with open(stats_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['frame', 'detections', 'inference_time'])
                writer.writeheader()
                writer.writerows(stats)
            print(f"\nStatistics saved to: {stats_path}")
    
    def detect_camera(
        self,
        camera_id: int = 0,
        output_path: str = None,
        record_duration: int = None
    ):
        """
        Real-time detection from camera feed.
        
        Parameters:
        -----------
        camera_id : int
            Camera device ID (0 for default camera)
        output_path : str
            Path to save recorded video (optional)
        record_duration : int
            Maximum recording duration in seconds (None for unlimited)
        """
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open camera {camera_id}")
        
        # Get camera properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30  # Default to 30 if not available
        
        print(f"\nCamera {camera_id} opened")
        print(f"Resolution: {width}x{height}")
        print(f"Press 'q' to quit, 'p' to pause, 's' to save frame\n")
        
        # Setup video writer if output specified
        out = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Statistics
        frame_count = 0
        total_detections = 0
        start_time = time.time()
        paused = False
        
        try:
            while True:
                # Check duration limit
                if record_duration and (time.time() - start_time) > record_duration:
                    print(f"\nRecording duration limit reached ({record_duration}s)")
                    break
                
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("\nWARNING: Failed to read frame")
                        break
                    
                    # Detect
                    annotated, detections, inf_time = self.detect(frame)
                    
                    # Update statistics
                    frame_count += 1
                    num_detections = len(detections)
                    total_detections += num_detections
                    
                    # Add info overlay
                    cv2.putText(annotated, f"Frame: {frame_count}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated, f"People: {num_detections}", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(annotated, f"FPS: {1/inf_time:.1f}", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    if record_duration:
                        elapsed = time.time() - start_time
                        remaining = record_duration - elapsed
                        cv2.putText(annotated, f"Time: {remaining:.1f}s", 
                                   (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Write to output video
                    if out:
                        out.write(annotated)
                
                # Show window
                cv2.imshow('Real-time People Detection', annotated)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"{'Paused' if paused else 'Resumed'}")
                elif key == ord('s') and not paused:
                    # Save current frame
                    save_path = Path('results') / f'frame_{frame_count:06d}.jpg'
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_path), annotated)
                    print(f"Saved frame to: {save_path}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            # Print summary
            elapsed_time = time.time() - start_time
            print(f"\n" + "="*60)
            print("CAMERA DETECTION SUMMARY")
            print("="*60)
            print(f"Total frames: {frame_count}")
            print(f"Total detections: {total_detections}")

    def detect_camera_zed(
        self,
        camera_id: int = 0,
        output_path: str = None,
        record_duration: int = None,
        resolution: str = 'HD2K',
        fps: int = 15,
        mirror: bool = False
    ):
        """
        Real-time detection from ZED 2i camera feed.
        
        Parameters:
        -----------
        camera_id : int
            ZED camera device ID (0 for default camera)
        output_path : str
            Path to save recorded video (optional)
        record_duration : int
            Maximum recording duration in seconds (None for unlimited)
        resolution : str
            Camera resolution: 'HD2K' (2208x1242), 'HD1080' (1920x1080), 'HD720' (1280x720), 'VGA' (672x376)
        fps : int
            Camera FPS: 15 or 30 (default: 15)
        mirror : bool
            Whether to horizontally flip the image (default: False)
        """
        if not ZED_AVAILABLE:
            raise ImportError(
                "ZED SDK not available. Please install pyzed SDK.\n"
                "See docs/ZED_SETUP.md for installation instructions."
            )
        
        # Map resolution string to ZED SDK enum
        resolution_map = {
            'HD2K': sl.RESOLUTION.HD2K,      # 2208x1242
            'HD1080': sl.RESOLUTION.HD1080,  # 1920x1080
            'HD720': sl.RESOLUTION.HD720,    # 1280x720
            'VGA': sl.RESOLUTION.VGA         # 672x376
        }
        
        if resolution not in resolution_map:
            raise ValueError(f"Invalid resolution '{resolution}'. Choose from: {list(resolution_map.keys())}")
        
        if fps not in [15, 30, 60]:
            raise ValueError(f"Invalid FPS '{fps}'. ZED 2i supports: 15, 30, or 60")
        
        # Initialize ZED camera
        zed = sl.Camera()
        
        # Set configuration parameters
        init_params = sl.InitParameters()
        init_params.camera_resolution = resolution_map[resolution]
        init_params.camera_fps = fps
        init_params.depth_mode = sl.DEPTH_MODE.NONE  # We don't need depth for detection
        
        # Try to open the camera
        err = zed.open(init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise ValueError(f"Could not open ZED camera {camera_id}: {err}")
        
        # Get camera information
        camera_info = zed.get_camera_information()
        width = camera_info.camera_configuration.resolution.width
        height = camera_info.camera_configuration.resolution.height
        fps = camera_info.camera_configuration.fps
        
        print(f"\nZED Camera {camera_id} opened")
        print(f"Model: {camera_info.camera_model}")
        print(f"Resolution: {width}x{height} @ {fps}fps")
        print(f"Serial Number: {camera_info.serial_number}")
        print(f"Press 'q' to quit, 'p' to pause, 's' to save frame\n")
        
        # Create Mat objects to store images
        image = sl.Mat()
        
        # Setup video writer if output specified
        out = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Runtime parameters
        runtime_params = sl.RuntimeParameters()
        
        # Statistics
        frame_count = 0
        total_detections = 0
        start_time = time.time()
        paused = False
        
        try:
            while True:
                # Check duration limit
                if record_duration and (time.time() - start_time) > record_duration:
                    print(f"\nRecording duration limit reached ({record_duration}s)")
                    break
                
                if not paused:
                    # Grab frame from ZED camera
                    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                        # Retrieve left image
                        zed.retrieve_image(image, sl.VIEW.LEFT)
                        
                        # Convert to OpenCV format (numpy array)
                        frame = image.get_data()
                        
                        # Convert RGBA to RGB (ZED SDK returns RGBA)
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
                        
                        # Apply mirroring if enabled
                        if mirror:
                            frame = cv2.flip(frame, 1)  # 1 = horizontal flip
                        
                        # Detect
                        annotated, detections, inf_time = self.detect(frame)
                        
                        # Update statistics
                        frame_count += 1
                        num_detections = len(detections)
                        total_detections += num_detections
                        
                        # Add info overlay
                        cv2.putText(annotated, f"Frame: {frame_count}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(annotated, f"People: {num_detections}", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(annotated, f"FPS: {1/inf_time:.1f}", 
                                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        if record_duration:
                            elapsed = time.time() - start_time
                            remaining = record_duration - elapsed
                            cv2.putText(annotated, f"Time: {remaining:.1f}s", 
                                       (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # Write to output video
                        if out:
                            out.write(annotated)
                    else:
                        print("\nWARNING: Failed to grab frame from ZED camera")
                        break
                
                
                # Resize for display (maintain aspect ratio)
                display_width = 1280
                display_height = 720
                display_frame = cv2.resize(annotated, (display_width, display_height))
                
                # Show window
                cv2.imshow('Real-time People Detection (ZED)', display_frame)
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('p'):
                    paused = not paused
                    print(f"{'Paused' if paused else 'Resumed'}")
                elif key == ord('s') and not paused:
                    # Save current frame
                    save_path = Path('results') / f'zed_frame_{frame_count:06d}.jpg'
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_path), annotated)
                    print(f"Saved frame to: {save_path}")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            zed.close()
            if out:
                out.release()
            cv2.destroyAllWindows()
            
            # Print summary
            elapsed_time = time.time() - start_time
            print(f"\n" + "="*60)
            print("ZED CAMERA DETECTION SUMMARY")
            print("="*60)
            print(f"Total frames: {frame_count}")
            print(f"Total detections: {total_detections}")
            print(f"Average detections/frame: {total_detections/frame_count:.2f}" if frame_count > 0 else "N/A")
            print(f"Duration: {elapsed_time:.1f}s")
            print(f"Average FPS: {frame_count/elapsed_time:.1f}" if elapsed_time > 0 else "N/A")
            
            if output_path and out:
                print(f"\nOutput video saved to: {output_path}")

            print(f"Average detections/frame: {total_detections/frame_count:.2f}" if frame_count > 0 else "N/A")
            print(f"Duration: {elapsed_time:.1f}s")
            print(f"Average FPS: {frame_count/elapsed_time:.1f}" if elapsed_time > 0 else "N/A")
            
            if output_path and out:
                print(f"\nOutput video saved to: {output_path}")


def main():
    """Command-line interface for people detection."""
    parser = argparse.ArgumentParser(
        description='People detection with YOLO/RT-DETR - supports images, videos, and camera'
    )
    parser.add_argument(
        'input',
        type=str,
        nargs='?',
        help='Input image or video file (omit for camera mode)'
    )
    parser.add_argument(
        '--camera',
        action='store_true',
        help='Use camera input (default camera ID: 0)'
    )
    parser.add_argument(
        '--camera-id',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    parser.add_argument(
        '--duration',
        type=int,
        help='Recording duration in seconds (camera mode only)')
    parser.add_argument(
        '--zed',
        action='store_true',
        help='Use ZED 2i camera instead of regular webcam (requires ZED SDK)'
    )
    parser.add_argument(
        '--zed-resolution',
        type=str,
        default='HD2K',
        choices=['HD2K', 'HD1080', 'HD720', 'VGA'],
        help='ZED camera resolution (default: HD2K). HD2K=2208x1242, HD1080=1920x1080, HD720=1280x720, VGA=672x376'
    )
    parser.add_argument(
        '--zed-fps',
        type=int,
        default=15,
        choices=[15, 30, 60],
        help='ZED camera FPS (default: 15). Options: 15, 30, 60'
    )
    parser.add_argument(
        '--zed-mirror',
        action='store_true',
        help='Horizontally flip the ZED camera image (mirror mode)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        help='Output image file'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='models/yolov8s.pt',
        help='Model path (default: models/yolov8s.pt). Supports YOLOv8/v10 (.pt) and RT-DETR (models/rtdetr-l.pt, models/rtdetr-x.pt)'
    )
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['auto', 'yolo', 'rtdetr'],
        default='auto',
        help='Model architecture type (default: auto-detect from filename)'
    )
    parser.add_argument(
        '--preset',
        type=str,
        choices=['max_accuracy', 'balanced', 'real_time', 'ultra_accuracy', 'adaptive_smart', 'ensemble_max', 'yola_max'],
        help='Use optimized preset configuration:\n' +
             'Basic CLAHE Presets:\n' +
             '  max_accuracy: RT-DETR-X + CLAHE (best standard detection)\n' +
             '  balanced: YOLOv10m + CLAHE (good speed/accuracy balance)\n' +
             '  real_time: YOLOv8m + CLAHE (fastest speed)\n' +
             'Advanced Presets:\n' +
             '  ultra_accuracy: RT-DETR-X + Zero-DCE++ Sequential\n' +
             '  yola_max: RT-DETR-X + YOLA Enhancement (NeurIPS 2024)\n' +
             '  adaptive_smart: YOLOv8m + Adaptive enhancement\n' +
             '  ensemble_max: YOLOv8m + Multi-enhancement fusion'
    )
    parser.add_argument(
        '--zero-dce',
        action='store_true',
        help='Enable Zero-DCE++ enhancement (requires weights download)'
    )
    parser.add_argument(
        '--yola',
        action='store_true',
        help='Enable YOLA enhancement'
    )
    parser.add_argument(
        '--yola-weights',
        type=str,
        help='Path to YOLA pretrained weights'
    )
    parser.add_argument(
        '--hybrid-mode',
        type=str,
        choices=['sequential', 'adaptive', 'ensemble'],
        help='Hybrid detection mode:\n' +
             '  sequential: Zero-DCE++ → YOLO pipeline\n' +
             '  adaptive: Auto-select enhancement based on brightness\n' +
             '  ensemble: Multi-path enhancement fusion'
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
        help='Display result window'
    )
    parser.add_argument(
        '--no-window',
        action='store_true',
        help='Disable display window (video mode only)'
    )
    parser.add_argument(
        '--save-stats',
        action='store_true',
        help='Save detection statistics to CSV (video mode only)'
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
        model_type=args.model_type,
        use_clahe=args.clahe,
        clahe_clip_limit=args.clip_limit,
        clahe_tile_size=args.tile_size,
        conf_threshold=args.conf,
        device=args.device,
        preset=args.preset,
        yola_weights=args.yola_weights
    )
    
    # Determine input mode
    if args.camera:
        # Camera mode - check if ZED camera requested
        if args.zed:
            if not ZED_AVAILABLE:
                print("Error: ZED SDK not available. Please install pyzed SDK.")
                print("See docs/ZED_SETUP.md for installation instructions.")
                return 1
            # ZED camera mode
            detector.detect_camera_zed(
                camera_id=args.camera_id,
                output_path=args.output,
                record_duration=args.duration,
                resolution=args.zed_resolution,
                fps=args.zed_fps,
                mirror=args.zed_mirror
            )
        else:
            # Regular webcam mode
            detector.detect_camera(
                camera_id=args.camera_id,
                output_path=args.output,
                record_duration=args.duration
            )
    elif args.input:
        # Check if input is video or image
        input_path = Path(args.input)
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        
        if input_path.suffix.lower() in video_extensions:
            # Video mode
            detector.detect_video(
                video_path=str(input_path),
                output_path=args.output,
                show_window=not args.no_window,
                save_stats=args.save_stats
            )
        else:
            # Image mode
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
    else:
        print("Error: Please provide input (image/video file) or use --camera flag")
        parser.print_help()
        return 1
    
    return 0
    return 0


if __name__ == '__main__':
    exit(main())
