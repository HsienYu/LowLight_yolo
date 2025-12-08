"""
Validate YOLO format annotations.
Checks for common issues and generates a validation report.
"""

import argparse
from pathlib import Path
from collections import defaultdict
import sys


def validate_annotations(image_dir, label_dir, verbose=False):
    """
    Validate YOLO format annotations.
    
    Returns:
    --------
    dict : Validation results with statistics and errors
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    
    results = {
        'total_images': 0,
        'total_labels': 0,
        'matched': 0,
        'missing_labels': [],
        'extra_labels': [],
        'empty_labels': [],
        'invalid_format': [],
        'out_of_range': [],
        'total_boxes': 0,
        'boxes_per_image': [],
        'valid': True,
        'errors': []
    }
    
    # Find all images
    image_files = set()
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
        image_files.update(image_dir.glob(ext))
    
    results['total_images'] = len(image_files)
    
    # Find all label files
    label_files = set(label_dir.glob('*.txt'))
    results['total_labels'] = len(label_files)
    
    if verbose:
        print(f"Found {results['total_images']} images")
        print(f"Found {results['total_labels']} label files\n")
    
    # Check for missing/extra labels
    for img_path in image_files:
        label_path = label_dir / f"{img_path.stem}.txt"
        
        if label_path.exists():
            results['matched'] += 1
            
            # Validate label file content
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                # Empty file is OK (no people in image)
                if not lines or (len(lines) == 1 and lines[0].strip() == ''):
                    results['empty_labels'].append(str(label_path.name))
                    results['boxes_per_image'].append(0)
                    continue
                
                num_boxes = 0
                for i, line in enumerate(lines, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) != 5:
                        results['invalid_format'].append(
                            f"{label_path.name}:{i} - Expected 5 values, got {len(parts)}"
                        )
                        results['valid'] = False
                        continue
                    
                    # Check class_id
                    try:
                        class_id = int(parts[0])
                        if class_id != 0:
                            results['invalid_format'].append(
                                f"{label_path.name}:{i} - Invalid class_id {class_id} (should be 0)"
                            )
                            results['valid'] = False
                    except ValueError:
                        results['invalid_format'].append(
                            f"{label_path.name}:{i} - Invalid class_id '{parts[0]}'"
                        )
                        results['valid'] = False
                    
                    # Check coordinates
                    try:
                        x, y, w, h = map(float, parts[1:])
                        
                        # Check range [0, 1]
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            results['out_of_range'].append(
                                f"{label_path.name}:{i} - Coordinates out of range: {x:.3f} {y:.3f} {w:.3f} {h:.3f}"
                            )
                            results['valid'] = False
                        
                        # Check that box is valid (width/height > 0)
                        if w <= 0 or h <= 0:
                            results['invalid_format'].append(
                                f"{label_path.name}:{i} - Invalid box size: w={w:.3f} h={h:.3f}"
                            )
                            results['valid'] = False
                        
                        num_boxes += 1
                        results['total_boxes'] += 1
                        
                    except ValueError as e:
                        results['invalid_format'].append(
                            f"{label_path.name}:{i} - Invalid coordinates: {e}"
                        )
                        results['valid'] = False
                
                results['boxes_per_image'].append(num_boxes)
                
            except Exception as e:
                results['invalid_format'].append(
                    f"{label_path.name} - Error reading file: {e}"
                )
                results['valid'] = False
        else:
            results['missing_labels'].append(str(img_path.name))
    
    # Check for extra label files
    image_stems = {img.stem for img in image_files}
    for label_path in label_files:
        if label_path.stem not in image_stems:
            results['extra_labels'].append(str(label_path.name))
    
    # Calculate statistics
    if results['boxes_per_image']:
        results['avg_boxes'] = sum(results['boxes_per_image']) / len(results['boxes_per_image'])
        results['min_boxes'] = min(results['boxes_per_image'])
        results['max_boxes'] = max(results['boxes_per_image'])
    else:
        results['avg_boxes'] = 0
        results['min_boxes'] = 0
        results['max_boxes'] = 0
    
    return results


def print_results(results):
    """Print validation results in a readable format."""
    print("\n" + "="*60)
    print("ANNOTATION VALIDATION REPORT")
    print("="*60)
    
    # Summary
    print("\nSUMMARY:")
    print(f"  Total images: {results['total_images']}")
    print(f"  Total label files: {results['total_labels']}")
    print(f"  Matched: {results['matched']}")
    print(f"  Empty labels (no people): {len(results['empty_labels'])}")
    
    # Errors
    print(f"\nERRORS:")
    if results['missing_labels']:
        print(f"  ⚠️  Missing labels: {len(results['missing_labels'])}")
    else:
        print(f"  ✓ No missing labels")
    
    if results['extra_labels']:
        print(f"  ⚠️  Extra labels: {len(results['extra_labels'])}")
    else:
        print(f"  ✓ No extra labels")
    
    if results['invalid_format']:
        print(f"  ✗ Invalid format: {len(results['invalid_format'])} issues")
    else:
        print(f"  ✓ All labels have valid format")
    
    if results['out_of_range']:
        print(f"  ✗ Out of range: {len(results['out_of_range'])} issues")
    else:
        print(f"  ✓ All coordinates in valid range")
    
    # Statistics
    print(f"\nSTATISTICS:")
    print(f"  Total bounding boxes: {results['total_boxes']}")
    print(f"  Average boxes per image: {results['avg_boxes']:.2f}")
    print(f"  Min boxes per image: {results['min_boxes']}")
    print(f"  Max boxes per image: {results['max_boxes']}")
    
    # Details
    if results['missing_labels']:
        print(f"\nMISSING LABELS (first 10):")
        for label in results['missing_labels'][:10]:
            print(f"  - {label}")
        if len(results['missing_labels']) > 10:
            print(f"  ... and {len(results['missing_labels']) - 10} more")
    
    if results['extra_labels']:
        print(f"\nEXTRA LABELS:")
        for label in results['extra_labels']:
            print(f"  - {label}")
    
    if results['invalid_format']:
        print(f"\nINVALID FORMAT (first 10):")
        for error in results['invalid_format'][:10]:
            print(f"  - {error}")
        if len(results['invalid_format']) > 10:
            print(f"  ... and {len(results['invalid_format']) - 10} more")
    
    if results['out_of_range']:
        print(f"\nOUT OF RANGE (first 10):")
        for error in results['out_of_range'][:10]:
            print(f"  - {error}")
        if len(results['out_of_range']) > 10:
            print(f"  ... and {len(results['out_of_range']) - 10} more")
    
    # Final verdict
    print("\n" + "="*60)
    if results['valid']:
        print("✓ VALIDATION PASSED - Ready for dataset preparation!")
    else:
        print("✗ VALIDATION FAILED - Please fix errors above")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Validate YOLO format annotations'
    )
    parser.add_argument(
        'image_dir',
        help='Directory containing images'
    )
    parser.add_argument(
        'label_dir',
        help='Directory containing label files'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    results = validate_annotations(args.image_dir, args.label_dir, args.verbose)
    print_results(results)
    
    # Exit with error code if validation failed
    sys.exit(0 if results['valid'] else 1)


if __name__ == '__main__':
    main()
