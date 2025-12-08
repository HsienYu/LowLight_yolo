"""
Extract frames from ZED2i SVO (Stereolabs Video) files.
Supports extracting left/right camera images at specified intervals.

Requires ZED SDK Python API to be installed:
https://www.stereolabs.com/docs/app-development/python/install/
"""

import sys
from pathlib import Path
import argparse
from typing import Optional

try:
    import pyzed.sl as sl
    ZED_AVAILABLE = True
except ImportError:
    ZED_AVAILABLE = False
    print("⚠️  ZED SDK not installed. Please install from:")
    print("   https://www.stereolabs.com/developers/release/")


def extract_frames_from_svo(
    svo_file: str,
    output_dir: str,
    frame_interval: int = 30,
    max_frames: Optional[int] = None,
    camera_side: str = 'left',
    resolution: str = 'HD720',
    start_frame: int = 0
):
    """
    Extract frames from SVO file.
    
    Parameters:
    -----------
    svo_file : str
        Path to SVO file
    output_dir : str
        Directory to save extracted frames
    frame_interval : int
        Extract every Nth frame (default: 30 = ~1 fps at 30fps recording)
    max_frames : int, optional
        Maximum number of frames to extract (None = all)
    camera_side : str
        'left', 'right', or 'both' (default: 'left')
    resolution : str
        'HD2K', 'HD1080', 'HD720', 'VGA' (default: 'HD720')
    start_frame : int
        Frame number to start extraction from (default: 0)
    
    Returns:
    --------
    int : Number of frames extracted
    """
    if not ZED_AVAILABLE:
        raise ImportError("ZED SDK not available. Please install it first.")
    
    # Initialize ZED camera with SVO file
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_file))
    
    # Set resolution
    resolution_map = {
        'HD2K': sl.RESOLUTION.HD2K,
        'HD1080': sl.RESOLUTION.HD1080,
        'HD720': sl.RESOLUTION.HD720,
        'VGA': sl.RESOLUTION.VGA
    }
    init_params.camera_resolution = resolution_map.get(resolution, sl.RESOLUTION.HD720)
    
    # Disable real-time mode for SVO playback
    init_params.svo_real_time_mode = False
    
    # Create camera object
    zed = sl.Camera()
    
    # Open camera
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"❌ Error opening SVO file: {status}")
        zed.close()
        return 0
    
    # Get video information
    nb_frames = zed.get_svo_number_of_frames()
    fps = zed.get_camera_information().camera_configuration.fps
    
    print(f"📹 SVO File: {Path(svo_file).name}")
    print(f"   Total frames: {nb_frames}")
    print(f"   FPS: {fps}")
    print(f"   Duration: {nb_frames/fps:.1f}s")
    print(f"   Resolution: {resolution}")
    print(f"   Extracting: every {frame_interval} frames")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare image containers
    left_image = sl.Mat()
    right_image = sl.Mat()
    
    # Set SVO position to start frame
    if start_frame > 0:
        zed.set_svo_position(start_frame)
    
    # Extract frames
    extracted_count = 0
    frame_number = start_frame
    
    print(f"\n🎬 Extracting frames to: {output_path}")
    print("=" * 60)
    
    while True:
        # Check max frames limit
        if max_frames and extracted_count >= max_frames:
            print(f"\n✓ Reached max frames limit ({max_frames})")
            break
        
        # Grab frame
        err = zed.grab()
        
        if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            print(f"\n✓ Reached end of SVO file")
            break
        elif err != sl.ERROR_CODE.SUCCESS:
            print(f"\n⚠️  Error grabbing frame {frame_number}: {err}")
            frame_number += 1
            continue
        
        # Extract frame at interval
        if frame_number % frame_interval == 0:
            svo_position = zed.get_svo_position()
            timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()
            
            # Extract left camera
            if camera_side in ['left', 'both']:
                zed.retrieve_image(left_image, sl.VIEW.LEFT)
                left_filename = output_path / f"frame_{frame_number:06d}_left.jpg"
                left_image.write(str(left_filename))
            
            # Extract right camera
            if camera_side in ['right', 'both']:
                zed.retrieve_image(right_image, sl.VIEW.RIGHT)
                right_filename = output_path / f"frame_{frame_number:06d}_right.jpg"
                right_image.write(str(right_filename))
            
            extracted_count += 1
            
            # Progress indicator
            progress = (frame_number / nb_frames) * 100
            print(f"Frame {frame_number:6d}/{nb_frames} ({progress:5.1f}%) - "
                  f"Extracted: {extracted_count:4d} frames", end='\r')
        
        frame_number += 1
    
    # Cleanup
    zed.close()
    
    print(f"\n{'=' * 60}")
    print(f"✅ Extraction complete!")
    print(f"   Frames extracted: {extracted_count}")
    print(f"   Saved to: {output_path}")
    
    return extracted_count


def batch_extract_svo_files(
    input_dir: str,
    output_dir: str,
    **kwargs
):
    """
    Extract frames from all SVO files in a directory.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing SVO files
    output_dir : str
        Base directory for extracted frames
    **kwargs : dict
        Additional arguments passed to extract_frames_from_svo
    """
    input_path = Path(input_dir)
    svo_files = list(input_path.glob("*.svo")) + list(input_path.glob("*.svo2"))
    
    if not svo_files:
        print(f"❌ No SVO files found in: {input_dir}")
        return
    
    print(f"Found {len(svo_files)} SVO file(s)")
    print("=" * 60)
    
    total_extracted = 0
    
    for i, svo_file in enumerate(svo_files, 1):
        print(f"\n📼 Processing file {i}/{len(svo_files)}: {svo_file.name}")
        print("-" * 60)
        
        # Create subdirectory for this SVO file
        video_output_dir = Path(output_dir) / svo_file.stem
        
        try:
            count = extract_frames_from_svo(
                str(svo_file),
                str(video_output_dir),
                **kwargs
            )
            total_extracted += count
        except Exception as e:
            print(f"❌ Error processing {svo_file.name}: {e}")
            continue
    
    print("\n" + "=" * 60)
    print(f"✅ Batch extraction complete!")
    print(f"   Total files processed: {len(svo_files)}")
    print(f"   Total frames extracted: {total_extracted}")
    print(f"   Output directory: {output_dir}")


def main():
    """Command-line interface for SVO frame extraction."""
    parser = argparse.ArgumentParser(
        description='Extract frames from ZED2i SVO files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract every 30th frame (1 fps at 30fps recording)
  python extract_svo_frames.py data/videos/recording.svo -o data/images/

  # Extract every 10th frame, max 500 frames
  python extract_svo_frames.py data/videos/recording.svo -o data/images/ -i 10 -m 500

  # Extract both left and right cameras
  python extract_svo_frames.py data/videos/recording.svo -o data/images/ -s both

  # Process all SVO files in a directory
  python extract_svo_frames.py data/videos/ -o data/images/ --batch
        """
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Input SVO file or directory (if --batch)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output directory for extracted frames'
    )
    parser.add_argument(
        '-i', '--interval',
        type=int,
        default=30,
        help='Extract every Nth frame (default: 30 = ~1fps at 30fps)'
    )
    parser.add_argument(
        '-m', '--max-frames',
        type=int,
        help='Maximum number of frames to extract per video'
    )
    parser.add_argument(
        '-s', '--side',
        type=str,
        choices=['left', 'right', 'both'],
        default='left',
        help='Camera side to extract (default: left)'
    )
    parser.add_argument(
        '-r', '--resolution',
        type=str,
        choices=['HD2K', 'HD1080', 'HD720', 'VGA'],
        default='HD720',
        help='Output resolution (default: HD720)'
    )
    parser.add_argument(
        '--start',
        type=int,
        default=0,
        help='Start frame number (default: 0)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all SVO files in input directory'
    )
    
    args = parser.parse_args()
    
    # Check if ZED SDK is available
    if not ZED_AVAILABLE:
        print("\n❌ ZED SDK Python API is not installed!")
        print("\nTo install:")
        print("1. Download ZED SDK from: https://www.stereolabs.com/developers/release/")
        print("2. Install the SDK for your platform")
        print("3. The Python API should be installed automatically")
        print("\nFor more info: https://www.stereolabs.com/docs/app-development/python/install/")
        return 1
    
    # Extract frames
    try:
        if args.batch:
            batch_extract_svo_files(
                args.input,
                args.output,
                frame_interval=args.interval,
                max_frames=args.max_frames,
                camera_side=args.side,
                resolution=args.resolution,
                start_frame=args.start
            )
        else:
            extract_frames_from_svo(
                args.input,
                args.output,
                frame_interval=args.interval,
                max_frames=args.max_frames,
                camera_side=args.side,
                resolution=args.resolution,
                start_frame=args.start
            )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())
