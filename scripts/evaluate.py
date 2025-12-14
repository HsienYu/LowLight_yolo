#!/usr/bin/env python3
"""
Evaluation Script for Low-Light YOLO Models

Supports:
- mAP50, mAP50-95, precision, recall metrics
- Test-time augmentation (TTA) for reference ceiling
- Per-brightness bin analysis
- Comparison across multiple models

Usage:
    # Basic evaluation
    python scripts/evaluate.py --model runs/train/exp/weights/best.pt

    # Evaluate with TTA
    python scripts/evaluate.py --model best.pt --tta

    # Evaluate on specific split
    python scripts/evaluate.py --model best.pt --split test

    # Compare multiple models
    python scripts/evaluate.py --model model1.pt model2.pt --compare
"""

import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np
from ultralytics import YOLO


def evaluate_model(
    model_path: str,
    data_yaml: str = 'data/dataset.yaml',
    split: str = 'test',
    imgsz: int = 640,
    batch: int = 16,
    device: str = None,
    conf: float = 0.001,
    iou: float = 0.6,
    augment: bool = False,  # TTA
    verbose: bool = True,
    save_json: bool = False,
    project: str = 'runs/evaluate',
    name: str = None
) -> Dict:
    """
    Evaluate a trained YOLO model.
    
    Parameters:
    -----------
    model_path : str
        Path to trained model weights
    data_yaml : str
        Path to dataset YAML configuration
    split : str
        Dataset split to evaluate on ('val', 'test')
    imgsz : int
        Input image size
    batch : int
        Batch size
    device : str
        Device to use (None for auto)
    conf : float
        Confidence threshold for evaluation
    iou : float
        IoU threshold for NMS
    augment : bool
        Enable test-time augmentation (TTA)
    verbose : bool
        Verbose output
    save_json : bool
        Save results in COCO JSON format
    project : str
        Project directory for saving results
    name : str
        Experiment name
    
    Returns:
    --------
    Dict with evaluation metrics
    """
    
    # Auto-detect device
    if device is None:
        import torch
        if torch.backends.mps.is_available():
            device = 'mps'
        elif torch.cuda.is_available():
            device = 'cuda'
        else:
            device = 'cpu'
    
    print("=" * 60)
    print("MODEL EVALUATION")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Data: {data_yaml}")
    print(f"Split: {split}")
    print(f"Device: {device}")
    print(f"TTA: {'Enabled' if augment else 'Disabled'}")
    print("=" * 60)
    
    # Generate name if not provided
    if name is None:
        model_name = Path(model_path).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tta_suffix = '_tta' if augment else ''
        name = f'{model_name}_{split}{tta_suffix}_{timestamp}'
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)
    
    # Run evaluation
    print(f"\nEvaluating on {split} set...")
    results = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        batch=batch,
        device=device,
        conf=conf,
        iou=iou,
        augment=augment,
        verbose=verbose,
        save_json=save_json,
        project=project,
        name=name
    )
    
    # Extract metrics
    metrics = {
        'model': str(model_path),
        'split': split,
        'tta': augment,
        'mAP50': float(results.box.map50),
        'mAP50-95': float(results.box.map),
        'precision': float(results.box.mp),
        'recall': float(results.box.mr),
        'f1': 2 * (float(results.box.mp) * float(results.box.mr)) / 
              (float(results.box.mp) + float(results.box.mr) + 1e-6),
    }
    
    # Per-class metrics (for single class, same as overall)
    if hasattr(results.box, 'ap50'):
        metrics['ap50_person'] = float(results.box.ap50[0]) if len(results.box.ap50) > 0 else 0.0
    
    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"mAP@0.5:      {metrics['mAP50']:.4f}")
    print(f"mAP@0.5:0.95: {metrics['mAP50-95']:.4f}")
    print(f"Precision:    {metrics['precision']:.4f}")
    print(f"Recall:       {metrics['recall']:.4f}")
    print(f"F1 Score:     {metrics['f1']:.4f}")
    print("=" * 60)
    
    # Save metrics to JSON
    metrics_path = Path(project) / name / 'metrics.json'
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {metrics_path}")
    
    return metrics


def compare_models(
    model_paths: List[str],
    data_yaml: str = 'data/dataset.yaml',
    split: str = 'test',
    output_csv: str = None,
    **kwargs
) -> List[Dict]:
    """
    Compare multiple models on the same dataset.
    
    Returns:
    --------
    List of metric dictionaries
    """
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"Models: {len(model_paths)}")
    print(f"Split: {split}")
    print("=" * 60)
    
    all_metrics = []
    
    for i, model_path in enumerate(model_paths, 1):
        print(f"\n[{i}/{len(model_paths)}] Evaluating: {model_path}")
        metrics = evaluate_model(
            model_path=model_path,
            data_yaml=data_yaml,
            split=split,
            **kwargs
        )
        all_metrics.append(metrics)
    
    # Print comparison table
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Model':<40} {'mAP50':>10} {'mAP50-95':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 80)
    
    for m in all_metrics:
        model_name = Path(m['model']).stem[:38]
        print(f"{model_name:<40} {m['mAP50']:>10.4f} {m['mAP50-95']:>10.4f} "
              f"{m['recall']:>10.4f} {m['f1']:>10.4f}")
    
    print("=" * 80)
    
    # Save comparison to CSV
    if output_csv:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_metrics[0].keys())
            writer.writeheader()
            writer.writerows(all_metrics)
        
        print(f"\nComparison saved to: {output_csv}")
    
    return all_metrics


def evaluate_with_tta_comparison(
    model_path: str,
    data_yaml: str = 'data/dataset.yaml',
    split: str = 'test',
    **kwargs
) -> Dict:
    """
    Evaluate model with and without TTA to measure improvement.
    """
    print("=" * 60)
    print("TTA COMPARISON")
    print("=" * 60)
    
    # Without TTA
    print("\n[1/2] Evaluating WITHOUT TTA...")
    metrics_no_tta = evaluate_model(
        model_path=model_path,
        data_yaml=data_yaml,
        split=split,
        augment=False,
        name=f"{Path(model_path).stem}_no_tta",
        **kwargs
    )
    
    # With TTA
    print("\n[2/2] Evaluating WITH TTA...")
    metrics_tta = evaluate_model(
        model_path=model_path,
        data_yaml=data_yaml,
        split=split,
        augment=True,
        name=f"{Path(model_path).stem}_tta",
        **kwargs
    )
    
    # Calculate improvement
    improvement = {
        'mAP50_delta': metrics_tta['mAP50'] - metrics_no_tta['mAP50'],
        'mAP50-95_delta': metrics_tta['mAP50-95'] - metrics_no_tta['mAP50-95'],
        'recall_delta': metrics_tta['recall'] - metrics_no_tta['recall'],
    }
    
    print("\n" + "=" * 60)
    print("TTA IMPROVEMENT")
    print("=" * 60)
    print(f"mAP@0.5:      {metrics_no_tta['mAP50']:.4f} -> {metrics_tta['mAP50']:.4f} "
          f"({improvement['mAP50_delta']:+.4f})")
    print(f"mAP@0.5:0.95: {metrics_no_tta['mAP50-95']:.4f} -> {metrics_tta['mAP50-95']:.4f} "
          f"({improvement['mAP50-95_delta']:+.4f})")
    print(f"Recall:       {metrics_no_tta['recall']:.4f} -> {metrics_tta['recall']:.4f} "
          f"({improvement['recall_delta']:+.4f})")
    print("=" * 60)
    
    return {
        'no_tta': metrics_no_tta,
        'tta': metrics_tta,
        'improvement': improvement
    }


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate YOLO models for low-light people detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic evaluation
  python scripts/evaluate.py --model runs/train/exp/weights/best.pt

  # Evaluate with TTA
  python scripts/evaluate.py --model best.pt --tta

  # Compare TTA effect
  python scripts/evaluate.py --model best.pt --tta-compare

  # Compare multiple models
  python scripts/evaluate.py --model model1.pt model2.pt --compare

  # Evaluate on test split
  python scripts/evaluate.py --model best.pt --split test
        """
    )
    
    # Model(s)
    parser.add_argument(
        '--model', '-m',
        nargs='+',
        required=True,
        help='Model path(s) to evaluate'
    )
    parser.add_argument(
        '--data', '-d',
        default='data/dataset.yaml',
        help='Dataset YAML path (default: data/dataset.yaml)'
    )
    
    # Evaluation options
    parser.add_argument(
        '--split', '-s',
        choices=['val', 'test'],
        default='test',
        help='Dataset split (default: test)'
    )
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='Image size (default: 640)'
    )
    parser.add_argument(
        '--batch', '-b',
        type=int,
        default=16,
        help='Batch size (default: 16)'
    )
    parser.add_argument(
        '--device',
        choices=['mps', 'cuda', 'cpu'],
        help='Device (default: auto-detect)'
    )
    
    # TTA options
    parser.add_argument(
        '--tta',
        action='store_true',
        help='Enable test-time augmentation'
    )
    parser.add_argument(
        '--tta-compare',
        action='store_true',
        help='Compare with and without TTA'
    )
    
    # Comparison mode
    parser.add_argument(
        '--compare',
        action='store_true',
        help='Compare multiple models'
    )
    parser.add_argument(
        '--output-csv',
        help='Output CSV for comparison results'
    )
    
    # Output options
    parser.add_argument(
        '--project',
        default='runs/evaluate',
        help='Project directory (default: runs/evaluate)'
    )
    parser.add_argument(
        '--name',
        help='Experiment name'
    )
    parser.add_argument(
        '--save-json',
        action='store_true',
        help='Save results in COCO JSON format'
    )
    
    args = parser.parse_args()
    
    # Determine evaluation mode
    if args.tta_compare:
        # TTA comparison mode
        if len(args.model) > 1:
            print("Warning: TTA comparison uses only the first model")
        evaluate_with_tta_comparison(
            model_path=args.model[0],
            data_yaml=args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            save_json=args.save_json
        )
    elif args.compare or len(args.model) > 1:
        # Multi-model comparison mode
        compare_models(
            model_paths=args.model,
            data_yaml=args.data,
            split=args.split,
            output_csv=args.output_csv or f'runs/evaluate/comparison_{args.split}.csv',
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            save_json=args.save_json
        )
    else:
        # Single model evaluation
        evaluate_model(
            model_path=args.model[0],
            data_yaml=args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            augment=args.tta,
            project=args.project,
            name=args.name,
            save_json=args.save_json
        )


if __name__ == '__main__':
    main()
