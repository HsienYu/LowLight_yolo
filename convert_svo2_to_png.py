import pyzed.sl as sl
import cv2
import os
from pathlib import Path

def convert_svo2_to_png(svo_file, output_dir):
    """Convert SVO2 file to PNG images"""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize ZED camera
    zed = sl.Camera()
    
    # Set configuration parameters
    init_params = sl.InitParameters()
    init_params.set_from_svo_file(str(svo_file))
    init_params.svo_real_time_mode = False  # Don't play in real-time
    
    # Open the SVO file
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"Error opening {svo_file}: {status}")
        return
    
    # Get SVO properties
    nb_frames = zed.get_svo_number_of_frames()
    print(f"Converting {svo_file.name}: {nb_frames} frames")
    
    # Create Mat objects to store images
    image = sl.Mat()
    
    # Base filename without extension
    base_name = svo_file.stem
    
    frame_count = 0
    
    # Loop through all frames
    while True:
        # Grab frame
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # Retrieve left image
            zed.retrieve_image(image, sl.VIEW.LEFT)
            
            # Convert to numpy array (OpenCV format)
            img_data = image.get_data()
            
            # Generate output filename
            output_file = os.path.join(output_dir, f"{base_name}_frame_{frame_count:06d}.png")
            
            # Save as PNG
            cv2.imwrite(output_file, cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR))
            
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"  Processed {frame_count}/{nb_frames} frames")
        else:
            break
    
    print(f"  Total frames extracted: {frame_count}")
    
    # Close the camera
    zed.close()

def main():
    # Directory containing SVO2 files
    svo_dir = Path(".")
    output_dir = Path("data/images")
    
    # Find all SVO2 files
    svo_files = list(svo_dir.glob("*.svo2"))
    
    if not svo_files:
        print("No SVO2 files found in current directory")
        return
    
    print(f"Found {len(svo_files)} SVO2 files")
    print("-" * 50)
    
    # Convert each file
    for svo_file in svo_files:
        convert_svo2_to_png(svo_file, output_dir)
        print("-" * 50)
    
    print("Conversion complete!")

if __name__ == "__main__":
    main()
