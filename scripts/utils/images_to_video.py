#!/usr/bin/env python3
"""
Convert a sequence of images to a video file.
"""

import cv2
import argparse
from pathlib import Path
import re
from tqdm import tqdm

def extract_frame_number(filename):
    """Extract frame number from filename for sorting."""
    match = re.search(r'frame_(\d+)', str(filename))
    if match:
        return int(match.group(1))
    return 0

def images_to_video(
    image_dir,
    output_path,
    fps=30,
    codec='mp4v',
    pattern=None
):
    """
    Convert images to video.
    
    Args:
        image_dir: Directory containing images
        output_path: Output video path
        fps: Frames per second
        codec: Video codec (mp4v, avc1, etc.)
        pattern: Optional glob pattern to filter images
    """
    image_dir = Path(image_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Find images
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
    images = []
    
    if pattern:
        images = list(image_dir.glob(pattern))
    else:
        for ext in extensions:
            images.extend(image_dir.glob(ext))
            
    if not images:
        print(f"❌ No images found in {image_dir}")
        return
        
    # Sort images by frame number
    try:
        images.sort(key=lambda x: extract_frame_number(x.name))
    except Exception:
        print("⚠️ Could not sort by frame number, using alphanumeric sort")
        images.sort(key=lambda x: x.name)
        
    print(f"Found {len(images)} images")
    print(f"First image: {images[0].name}")
    print(f"Last image: {images[-1].name}")
    
    # Read first image to get dimensions
    first_frame = cv2.imread(str(images[0]))
    if first_frame is None:
        print(f"❌ Could not read first image: {images[0]}")
        return
        
    height, width, layers = first_frame.shape
    size = (width, height)
    print(f"Video size: {width}x{height} @ {fps}fps")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(str(output_path), fourcc, fps, size)
    
    if not out.isOpened():
        print("❌ Could not open video writer. Try a different codec.")
        return
    
    # Write frames
    for img_path in tqdm(images, desc="Creating video"):
        frame = cv2.imread(str(img_path))
        if frame is None:
            print(f"⚠️ Skipping unreadable frame: {img_path.name}")
            continue
            
        # Ensure consistent size
        if frame.shape[:2] != (height, width):
            frame = cv2.resize(frame, size)
            
        out.write(frame)
        
    out.release()
    print(f"\n✅ Video saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Convert images to video')
    parser.add_argument('input_dir', help='Directory containing images')
    parser.add_argument('-o', '--output', default='data/videos/output.mp4',
                        help='Output video path')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second (default: 30)')
    parser.add_argument('--codec', type=str, default='mp4v',
                        help='Video codec (default: mp4v)')
    parser.add_argument('--pattern', type=str,
                        help='Glob pattern filter (e.g., "*frame*.png")')
    
    args = parser.parse_args()
    
    images_to_video(
        args.input_dir,
        args.output,
        args.fps,
        args.codec,
        args.pattern
    )

if __name__ == '__main__':
    main()
