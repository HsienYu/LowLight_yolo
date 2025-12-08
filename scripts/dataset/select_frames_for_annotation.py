"""
Select representative frames for manual annotation.
Creates a stratified sample of ~500 frames covering different scenarios.
"""

import csv
import random
import shutil
from pathlib import Path
from collections import defaultdict
import argparse


def load_detection_results(csv_path):
    """Load detection results from CSV."""
    results = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                results.append({
                    'image': row['image'],
                    'path': row['path'],
                    'people_detected': int(row['people_detected']),
                    'detections': row['detections'],
                    'confidences': [
                        float(d.split(':')[1]) 
                        for d in row['detections'].split('; ') 
                        if ':' in d
                    ]
                })
            except:
                pass
    return results


def get_video_source(filename):
    """Extract video ID from filename."""
    return filename.split('_frame_')[0]


def get_avg_confidence(result):
    """Get average confidence for a frame."""
    if not result['confidences']:
        return 0.0
    return sum(result['confidences']) / len(result['confidences'])


def select_frames(results, target_total=500):
    """
    Select representative frames for annotation.
    
    Strategy:
    1. All top crowded scenes (12 people)
    2. 50 frames from high-density video (19-37-43)
    3. 200 frames from main video (19-08-03)
    4. 100 low-confidence frames (<0.40)
    5. 100 zero-detection frames
    6. 50 frames from minority videos
    """
    
    selected = []
    used_images = set()
    
    # Group by video source
    by_video = defaultdict(list)
    for r in results:
        video = get_video_source(r['image'])
        by_video[video].append(r)
    
    print(f"Total frames: {len(results)}")
    print(f"Video sources: {list(by_video.keys())}")
    print()
    
    # 1. All frames with 12 people (max crowd)
    print("1. Selecting max crowd scenes (12 people)...")
    max_crowd = [r for r in results if r['people_detected'] == 12]
    for r in max_crowd:
        selected.append(r)
        used_images.add(r['image'])
    print(f"   Selected: {len(max_crowd)} frames")
    
    # 2. Random 50 from video 19-37-43 (high density)
    print("\n2. Selecting from high-density video (19-37-43)...")
    video_19_37_43 = [
        r for r in by_video['HD2K_SN34500148_19-37-43'] 
        if r['image'] not in used_images
    ]
    sample_19_37_43 = random.sample(video_19_37_43, min(50, len(video_19_37_43)))
    for r in sample_19_37_43:
        selected.append(r)
        used_images.add(r['image'])
    print(f"   Selected: {len(sample_19_37_43)} frames")
    
    # 3. Random 200 from video 19-08-03 (main dataset)
    print("\n3. Selecting from main video (19-08-03)...")
    video_19_08_03 = [
        r for r in by_video['HD2K_SN34500148_19-08-03'] 
        if r['image'] not in used_images
    ]
    sample_19_08_03 = random.sample(video_19_08_03, min(200, len(video_19_08_03)))
    for r in sample_19_08_03:
        selected.append(r)
        used_images.add(r['image'])
    print(f"   Selected: {len(sample_19_08_03)} frames")
    
    # 4. 100 low-confidence frames (<0.40 avg confidence)
    print("\n4. Selecting low-confidence frames (<0.40)...")
    low_conf = [
        r for r in results 
        if r['image'] not in used_images and get_avg_confidence(r) < 0.40
    ]
    sample_low_conf = random.sample(low_conf, min(100, len(low_conf)))
    for r in sample_low_conf:
        selected.append(r)
        used_images.add(r['image'])
    print(f"   Selected: {len(sample_low_conf)} frames")
    
    # 5. 100 zero-detection frames
    print("\n5. Selecting zero-detection frames...")
    zero_det = [
        r for r in results 
        if r['image'] not in used_images and r['people_detected'] == 0
    ]
    sample_zero = random.sample(zero_det, min(100, len(zero_det)))
    for r in sample_zero:
        selected.append(r)
        used_images.add(r['image'])
    print(f"   Selected: {len(sample_zero)} frames")
    
    # 6. 25 each from minority videos (19-37-06, 21-32-32)
    print("\n6. Selecting from minority videos...")
    for video_name in ['HD2K_SN34500148_19-37-06', 'HD2K_SN34500148_21-32-32']:
        if video_name in by_video:
            minority = [
                r for r in by_video[video_name] 
                if r['image'] not in used_images
            ]
            sample_minority = random.sample(minority, min(25, len(minority)))
            for r in sample_minority:
                selected.append(r)
                used_images.add(r['image'])
            print(f"   {video_name}: {len(sample_minority)} frames")
    
    # Fill remaining to reach target if needed
    remaining = target_total - len(selected)
    if remaining > 0:
        print(f"\n7. Filling remaining {remaining} slots randomly...")
        unused = [r for r in results if r['image'] not in used_images]
        if unused:
            fill = random.sample(unused, min(remaining, len(unused)))
            for r in fill:
                selected.append(r)
                used_images.add(r['image'])
            print(f"   Selected: {len(fill)} frames")
    
    return selected


def copy_selected_frames(selected, source_dir, output_dir):
    """Copy selected frames to output directory."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    source_dir = Path(source_dir)
    
    print(f"\nCopying {len(selected)} frames to {output_dir}...")
    copied = 0
    for result in selected:
        src = source_dir / result['image']
        dst = output_dir / result['image']
        
        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"   Warning: {src} not found")
    
    print(f"Copied {copied}/{len(selected)} frames")
    return copied


def save_selection_list(selected, output_file):
    """Save list of selected frames to text file."""
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        f.write("# Selected frames for annotation\n")
        f.write(f"# Total: {len(selected)} frames\n\n")
        for r in selected:
            f.write(f"{r['image']}\n")
    
    print(f"\nSaved frame list to: {output_file}")


def print_selection_summary(selected):
    """Print summary statistics of selected frames."""
    print("\n" + "="*60)
    print("SELECTION SUMMARY")
    print("="*60)
    
    # Count by video
    by_video = defaultdict(int)
    for r in selected:
        video = get_video_source(r['image'])
        by_video[video] += 1
    
    print("\nBy video source:")
    for video, count in sorted(by_video.items()):
        print(f"  {video}: {count} frames")
    
    # Count by people detected
    by_count = defaultdict(int)
    for r in selected:
        by_count[r['people_detected']] += 1
    
    print("\nBy people count:")
    for count in sorted(by_count.keys())[:15]:
        print(f"  {count} people: {by_count[count]} frames")
    
    # Confidence stats
    all_conf = []
    for r in selected:
        all_conf.extend(r['confidences'])
    
    if all_conf:
        import numpy as np
        print("\nConfidence statistics:")
        print(f"  Mean: {np.mean(all_conf):.3f}")
        print(f"  Median: {np.median(all_conf):.3f}")
        print(f"  Range: {np.min(all_conf):.3f} - {np.max(all_conf):.3f}")
    
    print(f"\nTotal selected: {len(selected)} frames")


def main():
    parser = argparse.ArgumentParser(
        description='Select representative frames for annotation'
    )
    parser.add_argument(
        'detection_csv',
        help='Path to detection results CSV (e.g., results/full_baseline_with_clahe.csv)'
    )
    parser.add_argument(
        '-n', '--num-frames',
        type=int,
        default=500,
        help='Target number of frames to select (default: 500)'
    )
    parser.add_argument(
        '--copy-to',
        help='Copy selected frames to this directory (optional)'
    )
    parser.add_argument(
        '--save-list',
        default='data/selected_for_annotation.txt',
        help='Save list of selected frames to this file (default: data/selected_for_annotation.txt)'
    )
    parser.add_argument(
        '--source-dir',
        default='data/images',
        help='Source directory containing images (default: data/images)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Load results
    print(f"Loading detection results from: {args.detection_csv}")
    results = load_detection_results(args.detection_csv)
    print(f"Loaded {len(results)} frames\n")
    
    # Select frames
    print("="*60)
    print("SELECTING FRAMES FOR ANNOTATION")
    print("="*60)
    selected = select_frames(results, target_total=args.num_frames)
    
    # Print summary
    print_selection_summary(selected)
    
    # Save list
    save_selection_list(selected, args.save_list)
    
    # Copy frames if requested
    if args.copy_to:
        copy_selected_frames(selected, args.source_dir, args.copy_to)
        print(f"\n✓ Selected frames copied to: {args.copy_to}")
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("1. Review the selected frames")
    print("2. Use annotation tool (LabelImg, Roboflow, CVAT)")
    print("3. Export annotations in YOLO format")
    print("4. Run dataset preparation script")
    print("5. Start fine-tuning")


if __name__ == '__main__':
    main()
