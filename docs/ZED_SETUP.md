# ZED SDK Setup Guide for MacOS

You have **4 SVO2 files** (~388MB total) from your ZED2i camera that need frame extraction.

## Option 1: Install ZED SDK (Recommended)

### Step 1: Download ZED SDK for MacOS

1. Visit: **https://www.stereolabs.com/developers/release/**
2. Download **ZED SDK for macOS** (ARM64 for Apple Silicon)
3. Current version: 4.x

### Step 2: Install ZED SDK

```bash
# After downloading, run the installer
# The installer will:
# 1. Install ZED SDK system-wide
# 2. Install Python API (pyzed)
# 3. Set up necessary libraries

# Follow the GUI installer prompts
```

### Step 3: Verify Installation

```bash
# Activate virtual environment
source venv/bin/activate

# Check if pyzed is available
python -c "import pyzed.sl as sl; print('✓ ZED SDK:', sl.get_sdk_version())"
```

### Step 4: Extract Frames from Your SVO Files

```bash
# Extract from all SVO files in batch (recommended)
python scripts/extract_svo_frames.py data/videos/ -o data/images/ --batch

# Or extract from single file
python scripts/extract_svo_frames.py data/videos/HD2K_SN34500148_19-08-03.svo2 -o data/images/

# Options:
# -i 30    : Extract every 30th frame (~1 fps at 30fps)
# -i 15    : Extract every 15th frame (~2 fps at 30fps)  
# -m 500   : Maximum 500 frames per video
# -s left  : Left camera only (default)
# -s both  : Both left and right cameras
```

---

## Option 2: Use ZED Depth Viewer (GUI Method)

If you prefer a GUI approach:

### Step 1: Install ZED SDK (as above)

### Step 2: Use Built-in Export Tool

```bash
# Open ZED Depth Viewer (installed with SDK)
ZED_Depth_Viewer

# Then:
# 1. File > Open SVO
# 2. Select your SVO file
# 3. Tools > Export > Images
# 4. Choose output directory
# 5. Set frame interval
# 6. Export
```

---

## Option 3: Convert SVO to Standard Video (Alternative)

If ZED SDK installation fails, you can convert SVO to MP4 first:

### Using ZED Tools (requires SDK):

```bash
# Convert SVO to MP4 (after SDK installation)
ZED_SVO_Editor -convert data/videos/HD2K_SN34500148_19-08-03.svo2 data/videos/output.mp4
```

### Then extract frames with ffmpeg:

```bash
# Install ffmpeg if needed
brew install ffmpeg

# Extract every 30th frame
ffmpeg -i data/videos/output.mp4 -vf "select='not(mod(n,30))'" -vsync 0 data/images/frame_%06d.jpg

# Or extract 1 frame per second
ffmpeg -i data/videos/output.mp4 -vf fps=1 data/images/frame_%06d.jpg
```

---

## Your SVO Files

```
data/videos/
├── HD2K_SN34500148_19-08-03.svo2  (250 MB) - Main recording
├── HD2K_SN34500148_19-37-06.svo2  (43 MB)  - Short clip
├── HD2K_SN34500148_19-37-43.svo2  (73 MB)  - Medium clip
└── HD2K_SN34500148_21-32-32.svo2  (22 MB)  - Short clip
```

**Estimated Output:**
- At 30 FPS, 250MB file ≈ 3-4 minutes ≈ 5,400-7,200 frames
- Extracting every 30th frame ≈ 180-240 images per large file
- Total from all files ≈ 300-400 images

---

## Recommended Workflow

### 1. **Install ZED SDK** (one-time setup)
   ```bash
   # Download and install from stereolabs.com
   ```

### 2. **Extract Frames** (5-10 minutes)
   ```bash
   # Process all videos at once
   source venv/bin/activate
   python scripts/extract_svo_frames.py data/videos/ -o data/images/ --batch -i 30
   ```

### 3. **Test Detection** (immediate)
   ```bash
   # Pick a frame and test
   python scripts/detect.py data/images/HD2K_SN34500148_19-08-03/frame_000030_left.jpg --compare
   ```

### 4. **Review Results** (manual)
   - Check `results/comparison/` for detection quality
   - Decide if more frames needed
   - Identify good frames for annotation

### 5. **Collect More if Needed**
   ```bash
   # If you need more diverse data
   # Extract with smaller interval:
   python scripts/extract_svo_frames.py data/videos/ -o data/images_more/ --batch -i 15
   ```

---

## Quick Start Commands

```bash
# 1. Activate environment
source venv/bin/activate

# 2. Extract frames (after ZED SDK installed)
python scripts/extract_svo_frames.py data/videos/ -o data/images/ --batch -i 30 -m 200

# 3. Test detection on extracted frames
python scripts/detect.py data/images/*/frame_*_left.jpg --compare

# 4. Review results
open results/comparison/
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'pyzed'"

**Solution:** Install ZED SDK from https://www.stereolabs.com/developers/release/

The Python API (pyzed) is included with the SDK installer.

### "Cannot open SVO file"

**Check:**
1. File is not corrupted: `ls -lh data/videos/*.svo2`
2. ZED SDK version supports SVO2 format (v3.x+)
3. File path is correct

### "Out of memory"

**Solutions:**
1. Extract fewer frames: use `-m 200` to limit
2. Lower resolution: use `-r HD720` instead of HD2K
3. Process files one at a time instead of batch

### "Frames are too dark"

**This is expected!** That's why we have CLAHE preprocessing.

After extraction, detection will use CLAHE automatically:
```bash
python scripts/detect.py data/images/frame_000030_left.jpg --clahe
```

---

## Next Steps After Frame Extraction

1. **Review extracted frames** - Check quality and lighting conditions
2. **Test baseline detection** - See how many people are detected
3. **Annotate good frames** - Use LabelImg for training data
4. **Fine-tune model** - Train on your specific data

---

## Support

- **ZED SDK Documentation**: https://www.stereolabs.com/docs/
- **Python API Reference**: https://www.stereolabs.com/docs/api/python/
- **Community Forum**: https://community.stereolabs.com/

