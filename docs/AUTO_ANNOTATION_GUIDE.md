# Auto-Annotation Workflow Guide

This guide shows you how to auto-annotate images first, then refine them manually in labelImg.

## Workflow Overview

1. **Auto-annotate** images using pre-trained YOLO model → generates `.txt` files
2. **Refine** annotations manually in labelImg → adjust/add/delete boxes
3. **Train** your model with the refined annotations

## Step 1: Auto-Annotate Images

### Basic Usage (Person Detection Only)
```bash
cd scripts
python auto_annotate.py ../data/to_annotate/ -o ../data/labels/ --person-only
```

### For Low-Light Images (with CLAHE preprocessing)
```bash
python auto_annotate.py ../data/to_annotate/ -o ../data/labels/ --clahe --person-only
```

### Lower Confidence Threshold (get more detections to review)
```bash
python auto_annotate.py ../data/to_annotate/ -o ../data/labels/ --conf 0.15 --person-only
```

### With Visualization (to preview results)
```bash
python auto_annotate.py ../data/to_annotate/ -o ../data/labels/ --viz --person-only
```

## Step 2: Refine Annotations in labelImg

After auto-annotation, open labelImg to review and adjust:

```bash
labelImg data/to_annotate/ data/classes.txt data/labels/
```

**In labelImg:**
- ✅ Review each auto-generated bounding box
- ✅ Adjust boxes that are slightly off
- ✅ Add missing persons that weren't detected
- ✅ Delete false positives (wrongly detected objects)
- ✅ Press 'D' to go to next image quickly

## Tips for Better Auto-Annotations

### 1. **Adjust Confidence Threshold**
- Lower threshold (0.15-0.20): More detections, more false positives to delete
- Higher threshold (0.30-0.40): Fewer detections, more misses to add manually
- Default (0.25): Good balance for most cases

### 2. **Use CLAHE for Low-Light Images**
```bash
python auto_annotate.py ../data/to_annotate/ -o ../data/labels/ --clahe --person-only
```

### 3. **Preview Results First**
Use `--viz` to save visualization images and review before opening labelImg:
```bash
python auto_annotate.py ../data/to_annotate/ -o ../data/labels/ --viz --person-only
```

### 4. **Multiple Classes**
If you need to detect multiple classes (not just person):
```bash
# Include person (0), bicycle (1), and car (2)
python auto_annotate.py ../data/to_annotate/ -o ../data/labels/ --classes "0,1,2"
```

## Common Issues & Solutions

### Issue: Too many false positives
**Solution:** Increase confidence threshold
```bash
python auto_annotate.py ../data/to_annotate/ -o ../data/labels/ --conf 0.35 --person-only
```

### Issue: Missing detections in dark areas
**Solution:** Use CLAHE preprocessing
```bash
python auto_annotate.py ../data/to_annotate/ -o ../data/labels/ --clahe --person-only
```

### Issue: Model not detecting well
**Solution:** Try a larger model (more accurate but slower)
```bash
# Instead of yolov8s.pt (small), use yolov8m.pt (medium) or yolov8l.pt (large)
python auto_annotate.py ../data/to_annotate/ -o ../data/labels/ -m yolov8m.pt --person-only
```

## File Structure

After auto-annotation, your structure will look like:
```
data/
├── to_annotate/          # Your images
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── labels/               # Auto-generated annotations
│   ├── image1.txt        # YOLO format: class x_center y_center width height
│   ├── image2.txt
│   └── ...
└── classes.txt           # Class names (person)
```

## Complete Example

```bash
# 1. Auto-annotate with CLAHE and lower confidence
cd scripts
python auto_annotate.py ../data/to_annotate/ -o ../data/labels/ \
    --clahe --conf 0.20 --person-only --viz

# 2. Review visualization images in data/labels/*_viz.jpg

# 3. Open labelImg to refine
cd ..
labelImg data/to_annotate/ data/classes.txt data/labels/

# 4. In labelImg: review, adjust, add, delete as needed
# 5. Save all changes (Ctrl+S or auto-save mode)
```

## Keyboard Shortcuts in labelImg

- **W**: Create new box
- **D**: Next image
- **A**: Previous image
- **Delete**: Delete selected box
- **Ctrl+S**: Save
- **Space**: Verify image
- **Ctrl+D**: Duplicate box

## Next Steps

After refining annotations:
1. Validate annotations: `python scripts/validate_annotations.py`
2. Prepare dataset: `python scripts/prepare_dataset.py`
3. Train your model with the refined annotations
