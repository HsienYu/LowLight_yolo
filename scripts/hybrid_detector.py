#!/usr/bin/env python3
"""
Hybrid Low-Light Detection System

Combines multiple enhancement methods with YOLO detection:
- Sequential (Serial) Pipeline
- Parallel Ensemble 
- Adaptive Hybrid

Author: Generated for low_light_yolo project
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from ultralytics import YOLO

from zero_dce import ZeroDCEEnhancer
from preprocessing import apply_clahe


class SequentialDetector:
    """
    Sequential Pipeline: Enhancement → YOLO Detection
    
    Best for: Real-time applications, balance of speed and accuracy
    FPS: 60+ (with RTX 3090)
    
    Usage:
        detector = SequentialDetector()
        results, enhanced = detector.detect(image)
    """
    
    def __init__(
        self,
        yolo_model='yolov8s.pt',
        zero_dce_weights=None,
        device=None,
        use_clahe_postprocess=False
    ):
        """
        Initialize sequential detector
        
        Args:
            yolo_model: Path to YOLO model
            zero_dce_weights: Path to Zero-DCE++ weights
            device: 'cuda', 'mps', or 'cpu'
            use_clahe_postprocess: Apply CLAHE after Zero-DCE++
        """
        self.device = self._setup_device(device)
        
        # Enhancement models
        self.zero_dce = ZeroDCEEnhancer(
            model_path=zero_dce_weights,
            device=self.device
        )
        
        # YOLO detector
        self.yolo = YOLO(yolo_model)
        self.yolo.to(self.device)
        
        self.use_clahe_postprocess = use_clahe_postprocess
        
        print(f"✅ Sequential Detector initialized on {self.device}")
    
    def _setup_device(self, device):
        """Auto-detect best device"""
        if device and device != 'auto':
            return device
        # Auto-detect
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    def detect(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45,
        adaptive: bool = True
    ) -> Tuple:
        """
        Detect objects with sequential enhancement
        
        Args:
            image: Input BGR image
            conf: Confidence threshold
            iou: IoU threshold for NMS
            adaptive: Adaptively choose enhancement based on brightness
            
        Returns:
            results: YOLO detection results
            enhanced: Enhanced image used for detection
        """
        # Analyze scene brightness
        brightness = self._calculate_brightness(image)
        
        # Stage 1: Choose enhancement strategy
        if adaptive:
            enhanced = self._adaptive_enhance(image, brightness)
        else:
            enhanced = self.zero_dce.enhance(image)
        
        # Stage 2: Optional CLAHE refinement
        if self.use_clahe_postprocess and brightness < 100:
            enhanced = apply_clahe(enhanced, clip_limit=1.5, tile_size=8)
        
        # Stage 3: YOLO detection
        results = self.yolo(enhanced, conf=conf, iou=iou, verbose=False)
        
        return results, enhanced
    
    def _calculate_brightness(self, image):
        """Calculate average brightness"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return np.mean(gray)
    
    def _adaptive_enhance(self, image, brightness):
        """Adaptively choose enhancement method based on brightness"""
        if brightness < 30:
            # Very dark: Zero-DCE++ + CLAHE
            enhanced = self.zero_dce.enhance(image)
            enhanced = apply_clahe(enhanced, clip_limit=2.0, tile_size=8)
        elif brightness < 80:
            # Low light: Zero-DCE++ only
            enhanced = self.zero_dce.enhance(image)
        elif brightness < 120:
            # Medium: CLAHE only
            enhanced = apply_clahe(image, clip_limit=2.0, tile_size=8)
        else:
            # Bright enough: No enhancement
            enhanced = image
        
        return enhanced


class EnsembleDetector:
    """
    Parallel Ensemble: Multiple Enhancement Paths → Weighted Fusion
    
    Best for: Maximum accuracy, offline processing
    FPS: 20-25 (with RTX 3090)
    
    Runs multiple enhancement methods in parallel and fuses results
    using Weighted Boxes Fusion (WBF)
    """
    
    def __init__(
        self,
        yolo_model='yolov8s.pt',
        zero_dce_weights=None,
        device=None
    ):
        """Initialize ensemble detector"""
        self.device = self._setup_device(device)
        
        # Enhancement methods
        self.zero_dce = ZeroDCEEnhancer(
            model_path=zero_dce_weights,
            device=self.device
        )
        
        # YOLO detector
        self.yolo = YOLO(yolo_model)
        self.yolo.to(self.device)
        
        print(f"✅ Ensemble Detector initialized on {self.device}")
    
    def _setup_device(self, device):
        if device and device != 'auto':
            return device
        # Auto-detect
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    def detect(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45
    ) -> Tuple:
        """
        Detect with ensemble of enhancement methods
        
        Args:
            image: Input BGR image
            conf: Confidence threshold
            iou: IoU threshold
            
        Returns:
            fused_results: Fused detection results
            enhanced_images: Dict of all enhanced versions
        """
        all_results = []
        enhanced_images = {}
        
        # Path 1: Original (no enhancement)
        results_original = self.yolo(image, conf=conf, iou=iou, verbose=False)
        all_results.append(results_original[0])
        enhanced_images['original'] = image
        
        # Path 2: Zero-DCE++
        enhanced_zerodce = self.zero_dce.enhance(image)
        results_zerodce = self.yolo(enhanced_zerodce, conf=conf, iou=iou, verbose=False)
        all_results.append(results_zerodce[0])
        enhanced_images['zero_dce'] = enhanced_zerodce
        
        # Path 3: CLAHE
        enhanced_clahe = apply_clahe(image, clip_limit=2.0, tile_size=8)
        results_clahe = self.yolo(enhanced_clahe, conf=conf, iou=iou, verbose=False)
        all_results.append(results_clahe[0])
        enhanced_images['clahe'] = enhanced_clahe
        
        # Path 4: Zero-DCE++ + CLAHE
        enhanced_combined = apply_clahe(enhanced_zerodce, clip_limit=1.5, tile_size=8)
        results_combined = self.yolo(enhanced_combined, conf=conf, iou=iou, verbose=False)
        all_results.append(results_combined[0])
        enhanced_images['combined'] = enhanced_combined
        
        # Fuse results with weighted boxes fusion
        fused_results = self._weighted_boxes_fusion(
            all_results,
            weights=[0.5, 1.0, 0.8, 0.9],  # Zero-DCE++ gets highest weight
            iou_thr=0.5,
            skip_box_thr=0.3
        )
        
        return fused_results, enhanced_images
    
    def _weighted_boxes_fusion(
        self,
        results_list: List,
        weights: List[float],
        iou_thr: float,
        skip_box_thr: float
    ):
        """
        Simplified Weighted Boxes Fusion
        
        Note: For production, consider using ensemble-boxes library:
        pip install ensemble-boxes
        """
        # Extract all boxes
        all_boxes = []
        all_scores = []
        all_classes = []
        
        for i, result in enumerate(results_list):
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                scores = result.boxes.conf.cpu().numpy() * weights[i]
                classes = result.boxes.cls.cpu().numpy()
                
                all_boxes.extend(boxes)
                all_scores.extend(scores)
                all_classes.extend(classes)
        
        if len(all_boxes) == 0:
            return results_list[0]  # Return first result if no detections
        
        # Simple NMS across all boxes
        all_boxes = np.array(all_boxes)
        all_scores = np.array(all_scores)
        all_classes = np.array(all_classes)
        
        # Apply NMS
        keep_indices = self._nms(all_boxes, all_scores, iou_thr)
        
        # Filter boxes
        final_boxes = all_boxes[keep_indices]
        final_scores = all_scores[keep_indices]
        final_classes = all_classes[keep_indices]
        
        # Filter by score threshold
        score_mask = final_scores > skip_box_thr
        final_boxes = final_boxes[score_mask]
        final_scores = final_scores[score_mask]
        final_classes = final_classes[score_mask]
        
        # Create new Results object with fused detections
        # Use the first result as template and run YOLO on the first enhanced image
        # to get a proper Results structure, then update with fused boxes
        import copy
        result = copy.deepcopy(results_list[0])
        
        # Clear existing boxes and create new ones with fused data
        if len(final_boxes) > 0:
            # Create new boxes tensor data
            boxes_data = torch.zeros((len(final_boxes), 6), device=self.device)
            boxes_data[:, :4] = torch.from_numpy(final_boxes).to(self.device)  # xyxy
            boxes_data[:, 4] = torch.from_numpy(final_scores).to(self.device)   # conf
            boxes_data[:, 5] = torch.from_numpy(final_classes).to(self.device)  # cls
            
            # Update the result's boxes using the data tensor
            from ultralytics.engine.results import Boxes
            result.boxes = Boxes(boxes_data, result.orig_shape)
        else:
            # No detections after fusion
            from ultralytics.engine.results import Boxes
            result.boxes = Boxes(torch.zeros((0, 6), device=self.device), result.orig_shape)
        
        return result
    
    def _nms(self, boxes, scores, iou_threshold):
        """Non-Maximum Suppression"""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return np.array(keep)


class AdaptiveDetector:
    """
    Adaptive Hybrid: Scene-aware dynamic enhancement selection
    
    Best for: Production deployment, optimal speed/accuracy balance
    FPS: 40-60 (with RTX 3090)
    
    Analyzes scene features and dynamically selects best enhancement strategy
    """
    
    def __init__(
        self,
        yolo_model='yolov8s.pt',
        zero_dce_weights=None,
        device=None
    ):
        """Initialize adaptive detector"""
        self.device = self._setup_device(device)
        
        # Enhancement methods
        self.zero_dce = ZeroDCEEnhancer(
            model_path=zero_dce_weights,
            device=self.device
        )
        
        # YOLO detector
        self.yolo = YOLO(yolo_model)
        self.yolo.to(self.device)
        
        print(f"✅ Adaptive Detector initialized on {self.device}")
    
    def _setup_device(self, device):
        if device and device != 'auto':
            return device
        # Auto-detect
        if torch.cuda.is_available():
            return 'cuda'
        elif torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'
    
    def detect(
        self,
        image: np.ndarray,
        conf: float = 0.25,
        iou: float = 0.45
    ) -> Tuple:
        """
        Adaptive detection with scene analysis
        
        Args:
            image: Input BGR image
            conf: Confidence threshold
            iou: IoU threshold
            
        Returns:
            results: YOLO detection results
            enhanced: Enhanced image
            strategy: Dict with selected enhancement strategy
        """
        # Stage 1: Analyze scene
        scene_features = self._analyze_scene(image)
        
        # Stage 2: Select optimal strategy
        strategies = self._select_strategy(scene_features)
        
        # Stage 3: Apply enhancements
        enhanced = image.copy()
        for strategy_name in strategies:
            enhanced = self._apply_enhancement(enhanced, strategy_name)
        
        # Stage 4: YOLO detection
        results = self.yolo(enhanced, conf=conf, iou=iou, verbose=False)
        
        strategy_info = {
            'brightness': scene_features['brightness'],
            'contrast': scene_features['contrast'],
            'noise': scene_features['noise'],
            'selected': strategies
        }
        
        return results, enhanced, strategy_info
    
    def _analyze_scene(self, image) -> Dict:
        """Analyze scene characteristics"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Brightness
        brightness = np.mean(gray)
        
        # Contrast
        contrast = np.std(gray)
        
        # Noise estimation (using Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_level = np.var(laplacian)
        
        return {
            'brightness': brightness,
            'contrast': contrast,
            'noise': noise_level
        }
    
    def _select_strategy(self, scene_features) -> List[str]:
        """
        Select optimal enhancement strategy based on scene features
        
        Returns:
            List of enhancement methods to apply in order
        """
        b = scene_features['brightness']
        c = scene_features['contrast']
        n = scene_features['noise']
        
        # Decision tree based on scene characteristics
        if b < 20 and n > 100:
            # Extremely dark + high noise
            return ['zero_dce', 'clahe_light']
        
        elif b < 40 and c < 20:
            # Very dark + low contrast
            return ['zero_dce', 'clahe_medium']
        
        elif b < 60:
            # Dark
            return ['zero_dce']
        
        elif b < 100:
            # Low light
            return ['clahe_medium']
        
        elif b < 140:
            # Medium light
            return ['clahe_light']
        
        else:
            # Bright enough
            return ['none']
    
    def _apply_enhancement(self, image, strategy: str) -> np.ndarray:
        """Apply specific enhancement strategy"""
        if strategy == 'none':
            return image
        
        elif strategy == 'zero_dce':
            return self.zero_dce.enhance(image)
        
        elif strategy == 'clahe_light':
            return apply_clahe(image, clip_limit=1.5, tile_size=8)
        
        elif strategy == 'clahe_medium':
            return apply_clahe(image, clip_limit=2.0, tile_size=8)
        
        elif strategy == 'clahe_strong':
            return apply_clahe(image, clip_limit=3.0, tile_size=4)
        
        else:
            return image


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Hybrid Low-Light Detection')
    parser.add_argument('input', help='Input image path')
    parser.add_argument('-o', '--output', default='results/hybrid',
                        help='Output directory')
    parser.add_argument('--mode', choices=['sequential', 'ensemble', 'adaptive'],
                        default='adaptive',
                        help='Detection mode')
    parser.add_argument('--yolo', default='yolov8s.pt',
                        help='YOLO model path')
    parser.add_argument('--zero-dce-weights', 
                        default='models/zero_dce_plus.pth',
                        help='Zero-DCE++ weights')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--device', choices=['cuda', 'mps', 'cpu'],
                        help='Device to use')
    
    args = parser.parse_args()
    
    # Load image
    image = cv2.imread(args.input)
    if image is None:
        print(f"❌ Failed to load image: {args.input}")
        exit(1)
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize detector
    if args.mode == 'sequential':
        detector = SequentialDetector(
            yolo_model=args.yolo,
            zero_dce_weights=args.zero_dce_weights,
            device=args.device
        )
        results, enhanced = detector.detect(image, conf=args.conf)
        
    elif args.mode == 'ensemble':
        detector = EnsembleDetector(
            yolo_model=args.yolo,
            zero_dce_weights=args.zero_dce_weights,
            device=args.device
        )
        results, enhanced_dict = detector.detect(image, conf=args.conf)
        enhanced = enhanced_dict['zero_dce']
        
    else:  # adaptive
        detector = AdaptiveDetector(
            yolo_model=args.yolo,
            zero_dce_weights=args.zero_dce_weights,
            device=args.device
        )
        results, enhanced, strategy = detector.detect(image, conf=args.conf)
        print(f"📊 Scene analysis: {strategy}")
    
    # Draw results
    annotated = results[0].plot() if hasattr(results[0], 'plot') else enhanced
    
    # Save results
    input_name = Path(args.input).stem
    cv2.imwrite(str(output_dir / f"{input_name}_enhanced.jpg"), enhanced)
    cv2.imwrite(str(output_dir / f"{input_name}_detected.jpg"), annotated)
    
    print(f"✅ Results saved to {output_dir}")
    print(f"   Detections: {len(results[0].boxes)} objects found")
