# ZED 2i Quick Start Guide

## Default Usage (HD2K @ 15fps)
```bash
python scripts/detect.py --camera --zed
```

## Common Configurations

### 2K @ 15fps (High Quality)
```bash
python scripts/detect.py --camera --zed --zed-resolution HD2K --zed-fps 15
```

### 2K @ 30fps (Balanced)
```bash
python scripts/detect.py --camera --zed --zed-resolution HD2K --zed-fps 30
```

### 1080p @ 30fps (Real-time)
```bash
python scripts/detect.py --camera --zed --zed-resolution HD1080 --zed-fps 30
```

### With Recording
```bash
python scripts/detect.py --camera --zed --zed-resolution HD2K -o output.mp4
```

### With Enhancement
```bash
python scripts/detect.py --camera --zed --zed-resolution HD2K --clahe
```

## Resolution Options
- `HD2K`: 2208x1242 (default)
- `HD1080`: 1920x1080
- `HD720`: 1280x720
- `VGA`: 672x376

## Mirror Mode
- `--zed-mirror`: Horizontally flip the camera image (off by default)
- Useful for making the view appear like a mirror

## FPS Options
- `15`: Default, best for high-res with processing
- `30`: Balanced
- `60`: Max speed (HD720/VGA only)

## Controls
- `q`: Quit
- `p`: Pause/Resume
- `s`: Save frame

## Display Settings
- Camera view window is automatically resized to 1280x720 for comfortable viewing
- Actual capture resolution remains as configured (HD2K, HD1080, HD720, or VGA)
- Saved frames and recorded videos maintain original capture resolution

### With Mirror Mode (Horizontal Flip)
```bash
python scripts/detect.py --camera --zed --zed-resolution HD2K --zed-mirror
```
