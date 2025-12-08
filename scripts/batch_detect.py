"""
Batch detection script to test multiple images and generate statistics.
"""

import sys
import csv
from pathlib import Path
from tqdm import tqdm
import argparse
import cv2

# Add scripts to path
sys.path.append(str(Path(__file__).parent))
from detect import PeopleDetector


def batch_detect(
    image_dir: str,
    output_csv: str = "results/detection_results.csv",
    model_path: str = 'yolov8s.pt',
    model_type: str = 'auto',
    use_clahe: bool = True,
    conf_threshold: float = 0.25,
    preset: str = None,
    save_images: bool = False,
    output_image_dir: str = None,
    max_images: int = None
):
    """
    Run detection on all images in a directory and save results to CSV.
    
    Args:
        save_images: If True, save annotated images with detection boxes
        output_image_dir: Directory to save annotated images (if save_images=True)
        max_images: Maximum number of images to process (None for all)
    """
    # Find all images
    image_dir = Path(image_dir)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_files.extend(image_dir.rglob(ext))
    
    if not image_files:
        print(f"No images found in: {image_dir}")
        return
    
    # Limit number of images if specified
    if max_images and max_images < len(image_files):
        image_files = image_files[:max_images]
        print(f"Limiting to first {max_images} images")
    
    print(f"Found {len(image_files)} images")
    if preset:
        print(f"Using preset: {preset}")
    else:
        print(f"Model: {model_path} (type: {model_type})")
        print(f"CLAHE: {'Enabled' if use_clahe else 'Disabled'}")
    print(f"Confidence threshold: {conf_threshold}")
    if preset:
        print(f"Preset: {preset}")
    if save_images:
        img_out = Path(output_image_dir) if output_image_dir else Path(output_csv).parent / 'images'
        img_out.mkdir(parents=True, exist_ok=True)
        print(f"Saving annotated images to: {img_out}")
    print("="*60)
    detector = PeopleDetector(
        model_path=model_path,
        model_type=model_type,
        use_clahe=use_clahe,
        conf_threshold=conf_threshold,
        preset=preset
    )
    
    # Results storage
    results = []
    total_detections = 0
    total_time = 0
    
    # Process each image
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            # detect_file returns (num_detections, inf_time)
            # Use detect() directly instead to get full detection info
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"Could not read image: {img_path}")
            
            annotated, detections, inf_time = detector.detect(image)
            
            num_people = len(detections)
            total_detections += num_people
            total_time += inf_time
            
            # Save annotated image if requested
            if save_images:
                img_out = Path(output_image_dir) if output_image_dir else Path(output_csv).parent / 'images'
                output_img_path = img_out / img_path.name
                cv2.imwrite(str(output_img_path), annotated)
            
            # Store results
            results.append({
                'image': img_path.name,
                'path': str(img_path),
                'people_detected': num_people,
                'inference_time': f"{inf_time:.3f}",
                'detections': '; '.join([
                    f"{d['class_name']}:{d['confidence']:.2f}"
                    for d in detections
                ])
            })
            
        except Exception as e:
            print(f"\nError processing {img_path.name}: {e}")
            results.append({
                'image': img_path.name,
                'path': str(img_path),
                'people_detected': 'ERROR',
                'inference_time': '0',
                'detections': str(e)
            })
    
    # Save results to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'image', 'path', 'people_detected', 'inference_time', 'detections'
        ])
        writer.writeheader()
        writer.writerows(results)
    
    # Print summary
    print("\n" + "="*60)
    print("DETECTION SUMMARY")
    print("="*60)
    print(f"Total images: {len(image_files)}")
    print(f"Total people detected: {total_detections}")
    print(f"Average detections per image: {total_detections/len(image_files):.2f}")
    print(f"Average inference time: {total_time/len(image_files):.3f}s")
    print(f"Images with detections: {sum(1 for r in results if r['people_detected'] not in ['ERROR', '0', 0])}")
    print(f"Images without detections: {sum(1 for r in results if r['people_detected'] in ['0', 0])}")
    print(f"\nResults saved to: {output_path}")
    
    # Show top detections
    print("\n" + "="*60)
    print("TOP FRAMES WITH MOST DETECTIONS")
    print("="*60)
    sorted_results = sorted(results, key=lambda x: int(x['people_detected']) if x['people_detected'] not in ['ERROR'] else 0, reverse=True)
    for i, result in enumerate(sorted_results[:10], 1):
        if result['people_detected'] in ['ERROR']:
            continue
        print(f"{i}. {result['image']}: {result['people_detected']} people")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Batch detection on all images in a directory'
    )
    parser.add_argument(
        'input_dir',
        type=str,
        help='Directory containing images'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='results/detection_results.csv',
        help='Output CSV file (default: results/detection_results.csv)'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='yolov8s.pt',
        help='Model path (default: yolov8s.pt). Supports YOLOv8/v10 and RT-DETR (rtdetr-l.pt, rtdetr-x.pt)'
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
        choices=['max_accuracy', 'balanced', 'real_time', 'ultra_accuracy', 'adaptive_smart', 'ensemble_max'],
        help='Use optimized preset configuration:\n' +
             'Basic CLAHE Presets:\n' +
             '  max_accuracy: RT-DETR-X + CLAHE (best standard detection)\n' +
             '  balanced: YOLOv10m + CLAHE (good speed/accuracy)\n' +
             '  real_time: YOLOv8m + CLAHE (fastest speed)\n' +
             'Advanced Zero-DCE++ Presets:\n' +
             '  ultra_accuracy: RT-DETR-X + Zero-DCE++ Sequential\n' +
             '  adaptive_smart: YOLOv8m + Adaptive enhancement\n' +
             '  ensemble_max: YOLOv8m + Multi-enhancement fusion'
    )
    parser.add_argument(
        '--clahe',
        action='store_true',
        help='Enable CLAHE preprocessing'
    )
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    parser.add_argument(
        '--save-images',
        action='store_true',
        help='Save annotated images with detection boxes'
    )
    parser.add_argument(
        '--output-image-dir',
        type=str,
        help='Directory to save annotated images (default: same as CSV output dir + /images)'
    )
    parser.add_argument(
        '--max-images',
        type=int,
        help='Maximum number of images to process (default: all)'
    )
    
    args = parser.parse_args()
    
    batch_detect(
        args.input_dir,
        output_csv=args.output,
        model_path=args.model,
        model_type=args.model_type,
        use_clahe=args.clahe,
        conf_threshold=args.conf,
        preset=args.preset,
        save_images=args.save_images,
        output_image_dir=args.output_image_dir,
        max_images=args.max_images
    )


if __name__ == '__main__':
    exit(main())
