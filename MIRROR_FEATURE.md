# Mirror Mode Feature for ZED Camera

## Summary
Added `--zed-mirror` command-line argument to enable horizontal flipping (mirroring) of the ZED camera image.

## Usage

### Enable Mirror Mode
```bash
python scripts/detect.py --camera --zed --zed-mirror
```

### Examples

#### Basic with Mirror
```bash
python scripts/detect.py --camera --zed --zed-resolution HD2K --zed-fps 15 --zed-mirror
```

#### With Recording and Mirror
```bash
python scripts/detect.py --camera --zed --zed-resolution HD1080 --zed-fps 30 --zed-mirror -o output.mp4
```

#### With Enhancement and Mirror
```bash
python scripts/detect.py --camera --zed --zed-resolution HD2K --clahe --zed-mirror
```

## Technical Details

### Implementation
The mirror function applies a horizontal flip to the camera frame immediately after RGBA to RGB conversion:

```python
# Apply mirroring if enabled
if mirror:
    frame = cv2.flip(frame, 1)  # 1 = horizontal flip
```

### Behavior
- **Default**: Mirror mode is OFF
- **When enabled**: Image is horizontally flipped before detection
- **Affects**: Live display, saved frames, and recorded videos
- **Performance**: Negligible impact (fast operation)

### Order of Operations
1. ZED camera captures frame (RGBA)
2. Convert RGBA → RGB
3. **Apply mirror flip (if enabled)**
4. Perform detection
5. Add annotations
6. Resize for display (1280x720)
7. Show window / save / record

## Use Cases

### Natural Mirror View
When the camera is facing you, mirror mode makes movements appear natural (like looking in a mirror):
- Move right → image moves right
- Without mirror: Move right → image moves left

### Self-Recording
Useful for recording yourself or demonstrations where natural movement direction is important.

### Theater/Stage Setup
When camera is positioned to face performers, mirror mode can provide a more intuitive view for monitoring.

## CLI Argument

```
--zed-mirror          Horizontally flip the ZED camera image (mirror mode)
```

- Type: Flag (boolean)
- Default: False (disabled)
- Affects: All ZED camera capture modes

## Notes

- Mirror mode is independent of resolution and FPS settings
- Can be combined with all enhancement methods (CLAHE, Zero-DCE++, YOLA)
- Works with all presets
- Recorded videos maintain the mirror flip if enabled
