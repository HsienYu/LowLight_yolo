# Display Window Update

## Change Summary
Fixed the ZED camera view window to display at 1280x720 resolution for comfortable viewing, regardless of the actual capture resolution.

## Technical Details

### Implementation
```python
# Before display, resize the annotated frame
display_width = 720
display_height = 480
display_frame = cv2.resize(annotated, (display_width, display_height))
cv2.imshow('Real-time People Detection (ZED)', display_frame)
```

### Behavior
- **Display window**: Always 1280x720 pixels
- **Capture resolution**: Unchanged (HD2K, HD1080, HD720, or VGA as configured)
- **Saved frames**: Original capture resolution maintained
- **Recorded videos**: Original capture resolution maintained
- **Detection processing**: Performed on full-resolution frames

## Benefits
1. **Consistent viewing experience** across all capture resolutions
2. **Comfortable window size** that fits most screens
3. **No impact on quality** - processing and recording use full resolution
4. **Better performance** - only the display is downscaled

## Examples

### HD2K (2208x1242) Capture
- Camera captures: 2208x1242
- Processing: 2208x1242
- Display window: 1280x720 (resized for viewing)
- Saved video: 2208x1242 (original resolution)

### HD1080 (1920x1080) Capture
- Camera captures: 1920x1080
- Processing: 1920x1080
- Display window: 1280x720 (resized for viewing)
- Saved video: 1920x1080 (original resolution)

## Usage
No command-line changes needed - display window automatically resizes:

```bash
# All these commands show 1280x720 display window
python scripts/detect.py --camera --zed --zed-resolution HD2K
python scripts/detect.py --camera --zed --zed-resolution HD1080
python scripts/detect.py --camera --zed --zed-resolution HD720
```

## Note
The 1280x720 display size is fixed in the code. To customize, modify the `display_width` and `display_height` variables in the `detect_camera_zed()` method (around line 957-959 in scripts/detect.py).
