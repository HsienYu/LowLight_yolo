"""
Prepare dataset for YOLOv8 training.
Splits annotated data into train/val/test sets and creates dataset.yaml.
"""

import argparse
import shutil
import random
from pathlib import Path
from collections import defaultdict
import yaml


def get_video_source(filename):
    """Extract video ID from filename."""
    return filename.split('_frame_')[0]


def stratified_split(image_files, labels_dir, train_ratio=0.7, val_ratio=0.2, seed=42):
    """
    Split data into train/val/test with stratification by video source.
    
    Returns:
    --------
    dict : {'train': [...], 'val': [...], 'test': [...]}
    """
    random.seed(seed)
    
    # Group by video source
    by_video = defaultdict(list)
    for img_path in image_files:
        video = get_video_source(img_path.name)
        by_video[video].append(img_path)
    
    splits = {'train': [], 'val': [], 'test': []}
    
    # Split each video source proportionally
    for video, images in by_video.items():
        random.shuffle(images)
        n = len(images)
        
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        splits['train'].extend(images[:n_train])
        splits['val'].extend(images[n_train:n_train + n_val])
        splits['test'].extend(images[n_train + n_val:])
    
    # Shuffle each split
    for split in splits.values():
        random.shuffle(split)
    
    return splits


def copy_files(splits, source_img_dir, source_label_dir, output_dir):
    """
    Copy images and labels to train/val/test directories.
    
    Directory structure:
    output_dir/
      ├── train/
      │   ├── images/
      │   └── labels/
      ├── val/
      │   ├── images/
      │   └── labels/
      └── test/
          ├── images/
          └── labels/
    """
    output_dir = Path(output_dir)
    source_img_dir = Path(source_img_dir)
    source_label_dir = Path(source_label_dir)
    
    stats = defaultdict(lambda: {'images': 0, 'labels': 0, 'boxes': 0})
    
    for split_name, image_files in splits.items():
        # Create directories
        img_dir = output_dir / split_name / 'images'
        label_dir = output_dir / split_name / 'labels'
        img_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCopying {split_name} set ({len(image_files)} images)...")
        
        for img_path in image_files:
            # Copy image
            shutil.copy2(img_path, img_dir / img_path.name)
            stats[split_name]['images'] += 1
            
            # Copy label
            label_path = source_label_dir / f"{img_path.stem}.txt"
            if label_path.exists():
                shutil.copy2(label_path, label_dir / label_path.name)
                stats[split_name]['labels'] += 1
                
                # Count boxes
                with open(label_path, 'r') as f:
                    lines = [line.strip() for line in f if line.strip()]
                    stats[split_name]['boxes'] += len(lines)
            else:
                # Create empty label file for images with no people
                (label_dir / f"{img_path.stem}.txt").touch()
                stats[split_name]['labels'] += 1
        
        print(f"  ✓ Copied {stats[split_name]['images']} images")
        print(f"  ✓ Copied {stats[split_name]['labels']} labels")
        print(f"  ✓ Total boxes: {stats[split_name]['boxes']}")
    
    return stats


def create_dataset_yaml(output_dir, project_root):
    """
    Create dataset.yaml configuration file for YOLOv8.
    """
    output_dir = Path(output_dir)
    project_root = Path(project_root)
    
    # Use relative paths from project root
    train_path = output_dir / 'train' / 'images'
    val_path = output_dir / 'val' / 'images'
    test_path = output_dir / 'test' / 'images'
    
    # Make paths relative to project root if possible
    try:
        train_rel = train_path.relative_to(project_root)
        val_rel = val_path.relative_to(project_root)
        test_rel = test_path.relative_to(project_root)
    except ValueError:
        # If not relative, use absolute paths
        train_rel = train_path.absolute()
        val_rel = val_path.absolute()
        test_rel = test_path.absolute()
    
    config = {
        'path': str(output_dir.absolute()),
        'train': str(train_rel),
        'val': str(val_rel),
        'test': str(test_rel),
        'names': {
            0: 'person'
        },
        'nc': 1  # number of classes
    }
    
    yaml_path = output_dir / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✓ Created dataset.yaml: {yaml_path}")
    return yaml_path


def print_summary(splits, stats):
    """Print dataset preparation summary."""
    print("\n" + "="*60)
    print("DATASET PREPARATION SUMMARY")
    print("="*60)
    
    total_images = sum(len(files) for files in splits.values())
    total_boxes = sum(s['boxes'] for s in stats.values())
    
    print(f"\nTotal images: {total_images}")
    print(f"Total bounding boxes: {total_boxes}")
    print(f"Average boxes per image: {total_boxes/total_images:.2f}")
    
    print("\nSplit distribution:")
    for split_name in ['train', 'val', 'test']:
        n_images = stats[split_name]['images']
        n_boxes = stats[split_name]['boxes']
        pct = n_images / total_images * 100
        avg_boxes = n_boxes / n_images if n_images > 0 else 0
        
        print(f"  {split_name:5s}: {n_images:4d} images ({pct:5.1f}%), "
              f"{n_boxes:5d} boxes (avg {avg_boxes:.2f}/img)")
    
    # Video source distribution
    print("\nBy video source:")
    by_video = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})
    for split_name, image_files in splits.items():
        for img_path in image_files:
            video = get_video_source(img_path.name)
            by_video[video][split_name] += 1
    
    for video in sorted(by_video.keys()):
        counts = by_video[video]
        total = sum(counts.values())
        print(f"  {video}:")
        print(f"    Train: {counts['train']:3d}, Val: {counts['val']:3d}, Test: {counts['test']:3d} "
              f"(Total: {total})")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Prepare annotated dataset for YOLOv8 training'
    )
    parser.add_argument(
        '--images',
        default='data/to_annotate',
        help='Directory containing annotated images (default: data/to_annotate)'
    )
    parser.add_argument(
        '--labels',
        default='data/labels',
        help='Directory containing label files (default: data/labels)'
    )
    parser.add_argument(
        '--output',
        default='data',
        help='Output directory for train/val/test splits (default: data)'
    )
    parser.add_argument(
        '--train-ratio',
        type=float,
        default=0.7,
        help='Training set ratio (default: 0.7)'
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='Validation set ratio (default: 0.2)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--project-root',
        default='.',
        help='Project root directory (default: current directory)'
    )
    
    args = parser.parse_args()
    
    # Validate ratios
    test_ratio = 1.0 - args.train_ratio - args.val_ratio
    if test_ratio <= 0:
        print("Error: train_ratio + val_ratio must be < 1.0")
        return 1
    
    print("="*60)
    print("DATASET PREPARATION")
    print("="*60)
    print(f"Images: {args.images}")
    print(f"Labels: {args.labels}")
    print(f"Output: {args.output}")
    print(f"Split ratios: Train={args.train_ratio:.1%}, Val={args.val_ratio:.1%}, "
          f"Test={test_ratio:.1%}")
    print(f"Random seed: {args.seed}")
    
    # Find images
    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    
    if not image_dir.exists():
        print(f"\nError: Image directory not found: {image_dir}")
        return 1
    
    if not label_dir.exists():
        print(f"\nError: Label directory not found: {label_dir}")
        return 1
    
    image_files = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.extend(image_dir.glob(ext))
    
    if not image_files:
        print(f"\nError: No images found in {image_dir}")
        return 1
    
    print(f"\nFound {len(image_files)} images")
    
    # Check for labels
    label_files = list(label_dir.glob('*.txt'))
    print(f"Found {len(label_files)} label files")
    
    if len(label_files) == 0:
        print("\nError: No label files found. Please annotate images first.")
        return 1
    
    # Stratified split
    print("\nPerforming stratified split by video source...")
    splits = stratified_split(
        image_files,
        label_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
    
    # Copy files
    print("\nCopying files to train/val/test directories...")
    stats = copy_files(splits, image_dir, label_dir, args.output)
    
    # Create dataset.yaml
    yaml_path = create_dataset_yaml(args.output, args.project_root)
    
    # Print summary
    print_summary(splits, stats)
    
    print("\n✓ Dataset preparation complete!")
    print(f"\nNext steps:")
    print(f"1. Review dataset.yaml: {yaml_path}")
    print(f"2. Start training: python scripts/train.py")
    
    return 0


if __name__ == '__main__':
    exit(main())
