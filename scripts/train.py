#!/usr/bin/env python3
"""
Training Script for Low-Light YOLO Models

Supports:
- Multiple model architectures (YOLOv8, YOLOv10)
- Mixing ratio experiments (raw, raw+CLAHE, raw+CLAHE+Zero-DCE++)
- Low-light optimized augmentation presets
- Automatic dataset configuration generation

Usage:
    # Basic training with raw data only
    python scripts/train.py --model yolov8m.pt --epochs 100

    # Training with CLAHE mixing (70:30 raw:clahe)
    python scripts/train.py --model yolov8m.pt --mix-clahe 0.3 --epochs 100

    # Training with preset configuration
    python scripts/train.py --preset low_light --model yolov8m.pt

    # Compare YOLOv8m vs YOLOv10m
    python scripts/train.py --model yolov8m.pt --name yolov8m_baseline
    python scripts/train.py --model yolov10m.pt --name yolov10m_baseline
"""

import argparse
import shutil
import yaml
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


# Training presets optimized for low-light detection
PRESETS = {
    'default': {
        'description': 'Default Ultralytics settings',
        'args': {}
    },
    'low_light': {
        'description': 'Optimized for low-light scenes (higher hsv_v, lower mosaic)',
        'args': {
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.5,  # Higher value variation for lighting robustness
            'mosaic': 0.2,  # Lower mosaic to preserve scene context
            'mixup': 0.0,
            'copy_paste': 0.0,
        }
    },
    'aggressive_aug': {
        'description': 'Aggressive augmentation for small datasets',
        'args': {
            'hsv_h': 0.02,
            'hsv_s': 0.8,
            'hsv_v': 0.6,
            'mosaic': 0.5,
            'mixup': 0.1,
            'degrees': 10.0,
            'translate': 0.2,
            'scale': 0.5,
            'fliplr': 0.5,
        }
    },
    'minimal_aug': {
        'description': 'Minimal augmentation for clean data',
        'args': {
            'hsv_h': 0.01,
            'hsv_s': 0.5,
            'hsv_v': 0.3,
            'mosaic': 0.0,
            'mixup': 0.0,
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.3,
        }
    }
}


def create_mixed_dataset_yaml(
    base_yaml: str,
    output_yaml: str,
    clahe_ratio: float = 0.0,
    zerodce_ratio: float = 0.0,
    clahe_dir: str = 'data_clahe',
    zerodce_dir: str = 'data_zerodce'
) -> str:
    """
    Create a dataset YAML that mixes raw and enhanced images.
    
    Ultralytics supports multiple train directories via list.
    We achieve mixing by including both raw and enhanced directories.
    
    Note: For true ratio control, you'd need to subsample directories.
    This implementation includes all images from specified directories.
    """
    with open(base_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    base_path = Path(config['path'])
    train_paths = [str(base_path / 'train' / 'images')]
    val_paths = [str(base_path / 'val' / 'images')]
    
    # Add CLAHE-enhanced images
    if clahe_ratio > 0:
        clahe_train = base_path.parent / clahe_dir / 'train' / 'images'
        clahe_val = base_path.parent / clahe_dir / 'val' / 'images'
        if clahe_train.exists():
            train_paths.append(str(clahe_train))
        if clahe_val.exists():
            val_paths.append(str(clahe_val))
    
    # Add Zero-DCE++ enhanced images
    if zerodce_ratio > 0:
        zerodce_train = base_path.parent / zerodce_dir / 'train' / 'images'
        zerodce_val = base_path.parent / zerodce_dir / 'val' / 'images'
        if zerodce_train.exists():
            train_paths.append(str(zerodce_train))
        if zerodce_val.exists():
            val_paths.append(str(zerodce_val))
    
    # Update config
    config['train'] = train_paths if len(train_paths) > 1 else train_paths[0]
    config['val'] = val_paths if len(val_paths) > 1 else val_paths[0]
    
    # Save new config
    output_path = Path(output_yaml)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print(f"Created mixed dataset config: {output_yaml}")
    print(f"  Train paths: {train_paths}")
    print(f"  Val paths: {val_paths}")
    
    return str(output_path)


def train(
    model_path: str = 'yolov8m.pt',
    data_yaml: str = 'data/dataset.yaml',
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = -1,  # Auto batch size
    device: str = None,
    project: str = 'runs/train',
    name: str = None,
    preset: str = 'low_light',
    mix_clahe: float = 0.0,
    mix_zerodce: float = 0.0,
    resume: bool = False,
    patience: int = 20,
    workers: int = 8,
    verbose: bool = True,
    **kwargs
):
    """
    Train YOLO model for low-light people detection.
    
    Parameters:
    -----------
    model_path : str
        Path to pretrained model (yolov8m.pt, yolov10m.pt, etc.)
    data_yaml : str
        Path to dataset YAML configuration
    epochs : int
        Number of training epochs
    imgsz : int
        Input image size
    batch : int
        Batch size (-1 for auto)
    device : str
        Device to use (None for auto, 'mps', 'cuda', 'cpu')
    project : str
        Project directory for saving results
    name : str
        Experiment name (auto-generated if None)
    preset : str
        Augmentation preset ('default', 'low_light', 'aggressive_aug', 'minimal_aug')
    mix_clahe : float
        Ratio of CLAHE-enhanced images to include (0.0-1.0)
    mix_zerodce : float
        Ratio of Zero-DCE++ enhanced images to include (0.0-1.0)
    resume : bool
        Resume from last checkpoint
    patience : int
        Early stopping patience
    workers : int
        Number of data loader workers
    verbose : bool
        Verbose output
    **kwargs
        Additional arguments passed to YOLO.train()
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
    print("LOW-LIGHT YOLO TRAINING")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Image size: {imgsz}")
    print(f"Preset: {preset}")
    if mix_clahe > 0:
        print(f"CLAHE mixing: {mix_clahe:.0%}")
    if mix_zerodce > 0:
        print(f"Zero-DCE++ mixing: {mix_zerodce:.0%}")
    print("=" * 60)
    
    # Create mixed dataset if needed
    if mix_clahe > 0 or mix_zerodce > 0:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mixed_yaml = f'data/dataset_mixed_{timestamp}.yaml'
        data_yaml = create_mixed_dataset_yaml(
            data_yaml, mixed_yaml,
            clahe_ratio=mix_clahe,
            zerodce_ratio=mix_zerodce
        )
    
    # Generate experiment name if not provided
    if name is None:
        model_name = Path(model_path).stem
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mix_suffix = ''
        if mix_clahe > 0:
            mix_suffix += f'_clahe{int(mix_clahe*100)}'
        if mix_zerodce > 0:
            mix_suffix += f'_zdce{int(mix_zerodce*100)}'
        name = f'{model_name}_{preset}{mix_suffix}_{timestamp}'
    
    # Load model
    print(f"\nLoading model: {model_path}")
    model = YOLO(model_path)
    
    # Get preset arguments
    train_args = PRESETS.get(preset, PRESETS['default'])['args'].copy()
    
    # Override with any additional kwargs
    train_args.update(kwargs)
    
    # Start training
    print(f"\nStarting training: {name}")
    print(f"Results will be saved to: {project}/{name}")
    
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        patience=patience,
        workers=workers,
        verbose=verbose,
        resume=resume,
        **train_args
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best model saved to: {project}/{name}/weights/best.pt")
    print(f"Last model saved to: {project}/{name}/weights/last.pt")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Train YOLO models for low-light people detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python scripts/train.py --model yolov8m.pt --epochs 100

  # Training with CLAHE mixing
  python scripts/train.py --model yolov8m.pt --mix-clahe 0.3

  # Training with low-light preset
  python scripts/train.py --model yolov8m.pt --preset low_light

  # Compare models
  python scripts/train.py --model yolov8m.pt --name yolov8m_exp1
  python scripts/train.py --model yolov10m.pt --name yolov10m_exp1

Presets:
  default       - Default Ultralytics settings
  low_light     - Optimized for low-light (higher hsv_v, lower mosaic)
  aggressive_aug- Heavy augmentation for small datasets
  minimal_aug   - Minimal augmentation for clean data
        """
    )
    
    # Model and data
    parser.add_argument(
        '--model', '-m',
        default='yolov8m.pt',
        help='Model path (default: yolov8m.pt)'
    )
    parser.add_argument(
        '--data', '-d',
        default='data/dataset.yaml',
        help='Dataset YAML path (default: data/dataset.yaml)'
    )
    
    # Training parameters
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=100,
        help='Number of epochs (default: 100)'
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
        default=-1,
        help='Batch size, -1 for auto (default: -1)'
    )
    parser.add_argument(
        '--device',
        choices=['mps', 'cuda', 'cpu'],
        help='Device (default: auto-detect)'
    )
    
    # Experiment configuration
    parser.add_argument(
        '--preset', '-p',
        choices=list(PRESETS.keys()),
        default='low_light',
        help='Augmentation preset (default: low_light)'
    )
    parser.add_argument(
        '--name', '-n',
        help='Experiment name (default: auto-generated)'
    )
    parser.add_argument(
        '--project',
        default='runs/train',
        help='Project directory (default: runs/train)'
    )
    
    # Mixing ratios
    parser.add_argument(
        '--mix-clahe',
        type=float,
        default=0.0,
        help='Include CLAHE-enhanced images (0.0-1.0, default: 0.0)'
    )
    parser.add_argument(
        '--mix-zerodce',
        type=float,
        default=0.0,
        help='Include Zero-DCE++ enhanced images (0.0-1.0, default: 0.0)'
    )
    
    # Other options
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from last checkpoint'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
        help='Early stopping patience (default: 20)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=8,
        help='Data loader workers (default: 8)'
    )
    
    args = parser.parse_args()
    
    # Run training
    train(
        model_path=args.model,
        data_yaml=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        preset=args.preset,
        mix_clahe=args.mix_clahe,
        mix_zerodce=args.mix_zerodce,
        resume=args.resume,
        patience=args.patience,
        workers=args.workers
    )


if __name__ == '__main__':
    main()
