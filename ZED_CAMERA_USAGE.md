# ZED 2i Camera Support for detect.py

## Overview
The `scripts/detect.py` now supports real-time detection using the ZED 2i stereo camera through the ZED SDK with configurable resolution and FPS.

## Requirements
- ZED SDK installed (pyzed module)
- See `docs/ZED_SETUP.md` for installation instructions

## Usage

### Basic ZED Camera Detection (2K @ 15fps - default)
```bash
python scripts/detect.py --camera --zed
```

### With 2K Resolution @ 30fps
```bash
python scripts/detect.py --camera --zed --zed-resolution HD2K --zed-fps 30
```

### With 1080p Resolution @ 15fps
```bash
python scripts/detect.py --camera --zed --zed-resolution HD1080 --zed-fps 15
```

### With Recording
```bash
python scripts/detect.py --camera --zed --zed-resolution HD2K --zed-fps 15 -o output/zed_detection.mp4
```

### With Time Limit (30 seconds)
```bash
python scripts/detect.py --camera --zed --duration 30
```

### With Enhanced Detection (CLAHE)
```bash
python scripts/detect.py --camera --zed --zed-resolution HD2K --clahe
```

### With Advanced Enhancement (YOLA)
```bash
python scripts/detect.py --camera --zed --zed-resolution HD2K --yola --preset yola_max
```

### Multiple ZED Cameras
If you have multiple ZED cameras, specify the camera ID:
```bash
python scripts/detect.py --camera --zed --camera-id 1
```

## Resolution Options

| Resolution | Size | Recommended FPS |
|------------|------|-----------------|
| `HD2K` | 2208x1242 | 15, 30 |
| `HD1080` | 1920x1080 | 15, 30 |
| `HD720` | 1280x720 | 15, 30, 60 |
| `VGA` | 672x376 | 15, 30, 60 |

**Defaults:** HD2K @ 15fps

## FPS Options

- **15 fps** - Best for high-resolution detection with lower bandwidth
- **30 fps** - Balanced performance (recommended for most use cases)
- **60 fps** - Maximum frame rate (only for HD720 and VGA)

## Features

### Supported Operations
- Real-time people detection from ZED 2i camera
- Configurable resolution (HD2K, HD1080, HD720, VGA)
- Configurable FPS (15, 30, 60)
- Video recording to MP4 format
- Frame-by-frame statistics (FPS, detection count)
- All enhancement methods (CLAHE, Zero-DCE++, YOLA)
- All preset configurations (max_accuracy, balanced, real_time, etc.)

### Controls (During Detection)
- **'q'** - Quit and show summary
- **'p'** - Pause/Resume detection
- **'s'** - Save current frame to `results/zed_frame_XXXXXX.jpg`

### Camera Configuration
- Display window: Fixed at 1280x720 for comfortable viewing (regardless of capture resolution)
- View: LEFT camera (matching SVO extraction workflow)
- Depth: Disabled (not needed for detection)

## Implementation Details

### Key Differences from Regular Webcam
1. Uses ZED SDK (`pyzed.sl`) instead of OpenCV's VideoCapture
2. Retrieves frames via `zed.grab()` and `zed.retrieve_image()`
3. Converts RGBA to RGB (ZED SDK returns RGBA format)
4. Provides camera model and serial number information
5. Supports multiple resolutions and frame rates

### Error Handling
- Gracefully handles missing ZED SDK with clear error message
- Camera initialization failures provide specific error codes
- Frame grab failures trigger warning and clean exit
- Invalid resolution/FPS parameters are validated with helpful messages

## Example Output (HD2K @ 15fps)
```
ZED Camera 0 opened
Model: ZED 2i
Resolution: 2208x1242 @ 15fps
Serial Number: 12345678
Press 'q' to quit, 'p' to pause, 's' to save frame

[Real-time detection window with overlays]

============================================================
ZED CAMERA DETECTION SUMMARY
============================================================
Total frames: 225
Total detections: 43
Average detections/frame: 0.19
Duration: 15.2s
Average FPS: 14.8

Output video saved to: output/zed_detection.mp4
```

## Performance Considerations

### Resolution vs Performance
- **HD2K (2208x1242)**: Highest quality, slower processing
- **HD1080 (1920x1080)**: Good balance for most applications
- **HD720 (1280x720)**: Faster processing, lower quality
- **VGA (672x376)**: Maximum speed, lowest quality

### FPS Recommendations
- **15 fps**: Recommended for HD2K with heavy processing (YOLA, Zero-DCE++)
- **30 fps**: Recommended for HD1080/HD720 with CLAHE
- **60 fps**: Only for HD720/VGA with minimal processing

## Common Usage Patterns

### High Quality Detection (Theater/Event)
```bash
python scripts/detect.py --camera --zed \
  --zed-resolution HD2K --zed-fps 15 \
  --clahe --preset max_accuracy \
  -o recordings/event.mp4
```

### Real-time Monitoring
```bash
python scripts/detect.py --camera --zed \
  --zed-resolution HD1080 --zed-fps 30 \
  --preset balanced
```

### Low-light Environment
```bash
python scripts/detect.py --camera --zed \
  --zed-resolution HD2K --zed-fps 15 \
  --yola --preset yola_max
```
