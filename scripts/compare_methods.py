#!/usr/bin/env python3
"""
Compare Enhancement Methods for Low-Light Detection

Evaluates and compares:
- Original (no enhancement)
- CLAHE
- Zero-DCE++
- Zero-DCE++ + CLAHE
- Sequential Detector
- Ensemble Detector
- Adaptive Detector

Metrics:
- Detection count
- Average confidence
- Inference time
- Visual quality (optional)
"""

import cv2
import numpy as np
import time
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
from ultralytics import YOLO

from preprocessing import apply_clahe
from zero_dce import ZeroDCEEnhancer
from hybrid_detector import SequentialDetector, EnsembleDetector, AdaptiveDetector


class MethodComparator:
    """
    Compare different enhancement methods for low-light detection
    """
    
    def __init__(self, yolo_model='yolov8s.pt', zero_dce_weights=None, device=None):
        """Initialize comparator"""
        self.device = device or ('cuda' if 'cuda' in str(device) else 'cpu')
        
        # Load models
        print("Loading models...")
        self.yolo = YOLO(yolo_model)
        self.zero_dce = ZeroDCEEnhancer(
            model_path=zero_dce_weights,
            device=self.device
        )
        
        # Initialize hybrid detectors
        self.sequential = SequentialDetector(
            yolo_model=yolo_model,
            zero_dce_weights=zero_dce_weights,
            device=self.device
        )
        self.ensemble = EnsembleDetector(
            yolo_model=yolo_model,
            zero_dce_weights=zero_dce_weights,
            device=self.device
        )
        self.adaptive = AdaptiveDetector(
            yolo_model=yolo_model,
            zero_dce_weights=zero_dce_weights,
            device=self.device
        )
        
        print("✅ All models loaded\n")
    
    def benchmark_single_image(
        self,
        image_path: str,
        conf: float = 0.25,
        warmup: int = 3
    ) -> Dict:
        """
        Benchmark all methods on a single image
        
        Args:
            image_path: Path to test image
            conf: Confidence threshold
            warmup: Number of warmup runs
            
        Returns:
            Dict with results for each method
        """
        print(f"Benchmarking: {Path(image_path).name}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        print(f"  Image brightness: {brightness:.1f}")
        
        results = {}
        
        # Warmup
        print(f"  Warming up ({warmup} runs)...")
        for _ in range(warmup):
            _ = self.yolo(image, conf=conf, verbose=False)
        
        # 1. Original (no enhancement)
        print("  Testing: Original")
        results['original'] = self._test_method(
            image, image, conf, method_name='original'
        )
        
        # 2. CLAHE
        print("  Testing: CLAHE")
        enhanced_clahe = apply_clahe(image, clip_limit=2.0, tile_size=8)
        results['clahe'] = self._test_method(
            image, enhanced_clahe, conf, method_name='clahe'
        )
        
        # 3. Zero-DCE++
        print("  Testing: Zero-DCE++")
        start = time.time()
        enhanced_zerodce = self.zero_dce.enhance(image)
        zerodce_time = time.time() - start
        results['zero_dce'] = self._test_method(
            image, enhanced_zerodce, conf, 
            method_name='zero_dce',
            enhancement_time=zerodce_time
        )
        
        # 4. Zero-DCE++ + CLAHE
        print("  Testing: Zero-DCE++ + CLAHE")
        enhanced_combined = apply_clahe(enhanced_zerodce, clip_limit=1.5, tile_size=8)
        results['zero_dce_clahe'] = self._test_method(
            image, enhanced_combined, conf,
            method_name='zero_dce_clahe',
            enhancement_time=zerodce_time + 0.002  # Approximate CLAHE time
        )
        
        # 5. Sequential Detector
        print("  Testing: Sequential Detector")
        start = time.time()
        seq_results, seq_enhanced = self.sequential.detect(image, conf=conf)
        seq_time = time.time() - start
        results['sequential'] = self._parse_results(
            seq_results[0], seq_time, 'sequential'
        )
        
        # 6. Adaptive Detector
        print("  Testing: Adaptive Detector")
        start = time.time()
        adp_results, adp_enhanced, adp_strategy = self.adaptive.detect(image, conf=conf)
        adp_time = time.time() - start
        results['adaptive'] = self._parse_results(
            adp_results[0], adp_time, 'adaptive'
        )
        results['adaptive']['strategy'] = adp_strategy['selected']
        
        # 7. Ensemble Detector (slower, optional)
        print("  Testing: Ensemble Detector")
        start = time.time()
        ens_results, ens_enhanced = self.ensemble.detect(image, conf=conf)
        ens_time = time.time() - start
        results['ensemble'] = self._parse_results(
            ens_results, ens_time, 'ensemble'
        )
        
        print("✅ Benchmark complete\n")
        return results
    
    def _test_method(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
        conf: float,
        method_name: str,
        enhancement_time: float = 0
    ) -> Dict:
        """Test a single enhancement method"""
        # Measure detection time
        start = time.time()
        results = self.yolo(enhanced, conf=conf, verbose=False)
        detection_time = time.time() - start
        
        return self._parse_results(
            results[0],
            detection_time + enhancement_time,
            method_name
        )
    
    def _parse_results(self, result, total_time: float, method_name: str) -> Dict:
        """Parse YOLO results into metrics"""
        boxes = result.boxes
        
        num_detections = len(boxes) if boxes is not None else 0
        avg_conf = float(boxes.conf.mean()) if num_detections > 0 else 0.0
        
        return {
            'method': method_name,
            'detections': num_detections,
            'avg_confidence': avg_conf,
            'inference_time_ms': total_time * 1000,
            'fps': 1.0 / total_time if total_time > 0 else 0
        }
    
    def compare_on_dataset(
        self,
        image_dir: str,
        output_csv: str = 'results/comparison.csv',
        conf: float = 0.25,
        max_images: int = None
    ):
        """
        Compare methods on entire dataset
        
        Args:
            image_dir: Directory containing test images
            output_csv: Path to save results CSV
            conf: Confidence threshold
            max_images: Maximum number of images to test (None for all)
        """
        image_dir = Path(image_dir)
        image_files = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        if max_images:
            image_files = image_files[:max_images]
        
        print(f"Testing on {len(image_files)} images...")
        
        all_results = []
        
        for img_path in image_files:
            try:
                results = self.benchmark_single_image(str(img_path), conf=conf, warmup=1)
                
                for method, metrics in results.items():
                    metrics['image'] = img_path.name
                    all_results.append(metrics)
                    
            except Exception as e:
                print(f"❌ Error processing {img_path.name}: {e}")
                continue
        
        # Save to CSV
        df = pd.DataFrame(all_results)
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        
        print(f"\n✅ Results saved to {output_csv}")
        
        # Print summary statistics
        self._print_summary(df)
        
        return df
    
    def _print_summary(self, df: pd.DataFrame):
        """Print summary statistics"""
        print("\n" + "=" * 70)
        print("SUMMARY STATISTICS")
        print("=" * 70)
        
        summary = df.groupby('method').agg({
            'detections': ['mean', 'std'],
            'avg_confidence': ['mean', 'std'],
            'inference_time_ms': ['mean', 'std'],
            'fps': 'mean'
        }).round(2)
        
        print(summary)
        print("=" * 70)
    
    def create_visual_comparison(
        self,
        image_path: str,
        output_path: str = 'results/visual_comparison.png',
        conf: float = 0.25
    ):
        """
        Create visual comparison figure
        
        Shows original + all enhancement methods with detection boxes
        """
        image = cv2.imread(str(image_path))
        
        # Get results for each method
        results = self.benchmark_single_image(image_path, conf=conf, warmup=0)
        
        # Prepare images
        images = {}
        images['original'] = image
        images['clahe'] = apply_clahe(image, clip_limit=2.0, tile_size=8)
        images['zero_dce'] = self.zero_dce.enhance(image)
        images['combined'] = apply_clahe(images['zero_dce'], clip_limit=1.5, tile_size=8)
        
        # Get detection results with boxes
        annotated = {}
        for name, img in images.items():
            det_results = self.yolo(img, conf=conf, verbose=False)
            annotated[name] = det_results[0].plot()
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        titles = [
            f"Original\n{results['original']['detections']} detections, {results['original']['avg_confidence']:.2f} conf",
            f"CLAHE\n{results['clahe']['detections']} detections, {results['clahe']['avg_confidence']:.2f} conf",
            f"Zero-DCE++\n{results['zero_dce']['detections']} detections, {results['zero_dce']['avg_confidence']:.2f} conf",
            f"Zero-DCE++ + CLAHE\n{results['zero_dce_clahe']['detections']} detections, {results['zero_dce_clahe']['avg_confidence']:.2f} conf"
        ]
        
        for ax, (name, img), title in zip(axes, annotated.items(), titles):
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_rgb)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.tight_layout()
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Visual comparison saved to {output_path}")
        plt.close()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Compare Enhancement Methods')
    parser.add_argument('input', help='Input image or directory')
    parser.add_argument('-o', '--output', default='results/comparison',
                        help='Output directory')
    parser.add_argument('--yolo', default='yolov8s.pt',
                        help='YOLO model path')
    parser.add_argument('--zero-dce-weights',
                        default='models/zero_dce_plus.pth',
                        help='Zero-DCE++ weights')
    parser.add_argument('--conf', type=float, default=0.25,
                        help='Confidence threshold')
    parser.add_argument('--device', choices=['cuda', 'mps', 'cpu'],
                        help='Device to use')
    parser.add_argument('--max-images', type=int,
                        help='Maximum images to test (for directories)')
    
    args = parser.parse_args()
    
    # Initialize comparator
    comparator = MethodComparator(
        yolo_model=args.yolo,
        zero_dce_weights=args.zero_dce_weights,
        device=args.device
    )
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        # Single image
        print("Single image comparison mode\n")
        
        # Benchmark
        results = comparator.benchmark_single_image(
            str(input_path),
            conf=args.conf
        )
        
        # Print results
        df = pd.DataFrame(results).T
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)
        print(df)
        
        # Create visual comparison
        comparator.create_visual_comparison(
            str(input_path),
            output_path=output_dir / f"{input_path.stem}_comparison.png",
            conf=args.conf
        )
        
    elif input_path.is_dir():
        # Dataset comparison
        print("Dataset comparison mode\n")
        
        df = comparator.compare_on_dataset(
            str(input_path),
            output_csv=output_dir / 'results.csv',
            conf=args.conf,
            max_images=args.max_images
        )
        
        # Create summary plots
        # TODO: Add visualization of aggregate results
    
    else:
        print(f"❌ Invalid input path: {input_path}")
