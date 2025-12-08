# Utility Scripts

Helper scripts for setup and data extraction.

## Files

### download_zero_dce_weights.py
Downloads or creates Zero-DCE++ model weights.

**Usage:**
```bash
# Show download instructions
python scripts/utils/download_zero_dce_weights.py

# Create dummy weights (for testing only)
python scripts/utils/download_zero_dce_weights.py --create-dummy
```

### extract_svo_frames.py
Extracts frames from ZED2i camera SVO files.

**Usage:**
```bash
# Extract frames from SVO files
python scripts/utils/extract_svo_frames.py data/videos/ -o data/images/ --batch -i 30

# Extract from single file
python scripts/utils/extract_svo_frames.py input.svo -o output_dir/ -i 30
```

### images_to_video.py
Converts a sequence of images to a video file.

**Usage:**
```bash
# Convert all images in directory to video
python scripts/utils/images_to_video.py data/images/ -o data/videos/output.mp4

# Specify FPS and pattern
python scripts/utils/images_to_video.py data/images/ -o output.mp4 --fps 60 --pattern "*frame*.png"
```

**Requirements:**
- ZED SDK must be installed (for extract_svo_frames)
- OpenCV (for images_to_video)
- See [ZED_SETUP.md](../../docs/ZED_SETUP.md) for detailed setup
