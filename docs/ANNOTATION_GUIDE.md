# Annotation Guide for Low-Light People Detection

This guide covers the complete annotation workflow for fine-tuning YOLOv8s on your ZED2i camera footage.

---

## Overview

**Goal**: Manually annotate 504 selected frames to create ground truth data  
**Class**: `person` (single class detection)  
**Format**: YOLO format (`.txt` files with normalized bounding boxes)  
**Estimated time**: 10-20 hours (1-2 minutes per frame)

---

## Selected Frames Summary

504 frames strategically selected from your 2,371-frame dataset:

| Category | Count | Purpose |
|----------|-------|---------|
| Max crowd (12 people) | 4 | Complex scenarios |
| High-density video | 58 | Challenging detection |
| Main dataset | 376 | Representative samples |
| Low confidence (<0.40) | Included | Improvement targets |
| Zero detections | 197 | Catch false negatives |

**Location**: `data/to_annotate/`  
**Frame list**: `data/selected_for_annotation.txt`

---

## Recommended Tools

### Option 1: LabelImg (Easiest for beginners)
**Best for**: Local annotation on Mac, simple interface

**Installation**:
```bash
pip install labelImg
```

**Launch**:
```bash
labelImg data/to_annotate/ data/classes.txt
```

**Setup**:
1. Click "Open Dir" → Select `data/to_annotate/`
2. Click "Change Save Dir" → Select `data/labels/` (will be created)
3. Select "PascalVOC" → Switch to "YOLO" format
4. Start annotating!

### Option 2: Roboflow (Best for teams/cloud)
**Best for**: Team collaboration, auto-labeling assistance, version control

**Workflow**:
1. Create account at [roboflow.com](https://roboflow.com)
2. Create new project → Object Detection → Single class (`person`)
3. Upload `data/to_annotate/` frames
4. Use Smart Polygon or SAM for assisted labeling
5. Export in "YOLOv8" format when complete

### Option 3: CVAT (Most features)
**Best for**: Advanced users, complex scenarios, video annotation

**Setup** (Docker):
```bash
docker-compose -f docker-compose.yml up -d
```

Access at `http://localhost:8080`

---

## Annotation Guidelines

### What to Annotate

✅ **DO annotate**:
- All visible people, even if partially visible
- People in shadows or low light
- Distant people (even if small)
- Partially occluded people (>30% visible)
- People at any angle/pose
- People walking, standing, sitting

❌ **DO NOT annotate**:
- Reflections or shadows (not actual people)
- People <10 pixels tall (too small to detect)
- People >70% occluded (not enough visible)
- Mannequins, statues, or drawings
- People in posters/screens

### Bounding Box Guidelines

**Tight boxes**: Include entire visible person, exclude empty space
```
Good:     [  👤  ]    Tight fit
Bad:      [    👤      ]    Too much padding
```

**Partial visibility**: Box only the visible parts
```
Occluded:  [👤]|wall    Don't extend box behind wall
```

**Groups**: Separate box for each person, even if overlapping
```
Crowd:  [👤] [👤] [👤]   Individual boxes
```

### Consistency Rules

1. **Include head to feet** (or visible portions if occluded)
2. **Exclude held objects** (bags, umbrellas) unless attached to body
3. **Ignore motion blur** - annotate as if person is still
4. **Low confidence** - if unsure if it's a person, mark it (helps model learn)

---

## LabelImg Workflow (Recommended)

### Setup (One-time)

1. Create classes file:
```bash
echo "person" > data/classes.txt
```

2. Launch LabelImg:
```bash
cd /Users/chenghsienyu/GitRepos/low_light_yolo
source venv/bin/activate
pip install labelImg
labelImg data/to_annotate/ data/classes.txt
```

### Annotation Steps

1. **Open LabelImg** → Should show first image from `data/to_annotate/`

2. **Set format** → Click "PascalVOC" button to switch to "YOLO"

3. **Change save directory**:
   - File → Change Save Dir
   - Select: `data/labels/`
   - (Create folder if it doesn't exist)

4. **Annotate current image**:
   - Press `W` or click "Create RectBox"
   - Click and drag to draw bounding box around person
   - Select "person" from class list
   - Repeat for all people in image

5. **Save and next**:
   - Press `Ctrl+S` to save (or click "Save")
   - Press `D` to move to next image
   - Continue until all 504 frames are done

### Keyboard Shortcuts (Speed up annotation)

| Key | Action |
|-----|--------|
| `W` | Create bounding box |
| `D` | Next image |
| `A` | Previous image |
| `Ctrl+S` | Save |
| `Del` | Delete selected box |
| `Ctrl+D` | Duplicate box |
| `↑↓←→` | Move box |
| `Ctrl+E` | Edit label |

### Quality Checks

Every 50 frames, check:
- [ ] All people annotated (no missed detections)
- [ ] Boxes are tight (no excess padding)
- [ ] No duplicate/overlapping boxes
- [ ] Saved to correct directory (`data/labels/`)

---

## Expected Output Format

For each image `frame_000123.png`, create `frame_000123.txt`:

```
0 0.5123 0.3456 0.1234 0.2345
0 0.7234 0.5678 0.0987 0.1567
```

**Format**: `class_id x_center y_center width height`
- `class_id`: Always `0` (person class)
- `x_center`: Normalized center X (0.0-1.0)
- `y_center`: Normalized center Y (0.0-1.0)
- `width`: Normalized box width (0.0-1.0)
- `height`: Normalized box height (0.0-1.0)

**Example** (image 640×480):
```
Person at pixel box: [100, 50, 200, 300]
↓ Normalized (divide by image size):
x_center = (100+200)÷2÷640 = 0.234
y_center = (50+300)÷2÷480 = 0.365
width = (200-100)÷640 = 0.156
height = (300-50)÷480 = 0.521
↓ YOLO format:
0 0.234 0.365 0.156 0.521
```

LabelImg does this conversion automatically!

---

## Quality Assurance

### Self-Review Checklist

After completing annotation:

1. **Completeness check**:
```bash
# Count annotated frames
ls data/labels/*.txt | wc -l
# Should be ~504 (one per image)
```

2. **Visual spot check**:
```bash
# Check 10 random frames
python scripts/visualize_annotations.py data/to_annotate/ data/labels/ --random 10
```

3. **Statistics check**:
```bash
python scripts/annotation_stats.py data/labels/
```

Expected output:
- Total boxes: ~1,600-2,000 (504 frames × 3.2 avg people)
- Avg boxes/image: 3.0-4.0
- Min/max boxes: 0-12

### Common Issues

**Issue**: Missing `.txt` files for some images  
**Fix**: Go back and annotate missing frames

**Issue**: Empty `.txt` files (0 bytes)  
**Fix**: These are frames with no people (intentional)

**Issue**: Very large box coordinates (>1.0)  
**Fix**: Ensure YOLO format is enabled, not PascalVOC

---

## Annotation Strategy (Time-saving tips)

### Batch Similar Frames
- Video frames are often consecutive → people in similar positions
- Use `Ctrl+D` to duplicate previous box and adjust slightly

### Prioritize Quality Over Speed
- 1 minute per frame with good quality > 30 seconds with mistakes
- Take breaks every hour to maintain focus

### Handle Edge Cases
- **Uncertain if person?** → Annotate it (model will learn)
- **Heavily occluded?** → Skip if <30% visible
- **Group/crowd?** → Individual boxes, overlapping OK
- **Reflection/glass?** → Only annotate real people, not reflections

### Save Frequently
- LabelImg autosaves, but press `Ctrl+S` often
- Periodically check `data/labels/` folder

---

## Validation After Annotation

Once all 504 frames are annotated:

```bash
# Run validation script
python scripts/validate_annotations.py data/to_annotate/ data/labels/

# Generate statistics report
python scripts/annotation_stats.py data/labels/ --save-report results/annotation_report.txt
```

Expected validation output:
```
✓ Found 504 images
✓ Found 504 annotation files
✓ All annotations in valid format
✓ No empty annotations (except frames with no people)
✓ Bounding boxes within valid range [0.0, 1.0]
✓ Average boxes per image: 3.2
✓ Total annotated people: 1,613

Ready for dataset preparation!
```

---

## Next Steps

After completing annotation:

1. ✅ **Validate annotations**:
```bash
python scripts/validate_annotations.py data/to_annotate/ data/labels/
```

2. ✅ **Prepare dataset**:
```bash
python scripts/prepare_dataset.py
```
This will:
- Split data into train/val/test (70/20/10)
- Create `dataset.yaml` configuration
- Copy images and labels to proper directories
- Generate dataset statistics

3. ✅ **Start fine-tuning**:
```bash
python scripts/train.py
```

---

## Troubleshooting

### LabelImg won't start
```bash
# Reinstall
pip uninstall labelImg
pip install labelImg

# Or use Python module directly
python -m labelImg
```

### Wrong save format (PascalVOC instead of YOLO)
- Click the "PascalVOC" button in LabelImg to toggle to "YOLO"
- Re-save all annotations

### Can't see images
- Check path: `data/to_annotate/` should contain 504 PNG files
- Try absolute path: `/Users/chenghsienyu/GitRepos/low_light_yolo/data/to_annotate/`

### Annotations not saving
- Check write permissions on `data/labels/` folder
- Try creating folder manually: `mkdir -p data/labels`

---

## Estimated Timeline

| Task | Time | Notes |
|------|------|-------|
| Tool setup | 15 min | One-time installation |
| First 50 frames | 2-3 hours | Learning curve |
| Next 200 frames | 5-7 hours | Getting faster |
| Last 254 frames | 5-7 hours | Experienced |
| **Total** | **12-17 hours** | Can split over 2-3 days |

**Tip**: Aim for 50-100 frames per session, take breaks!

---

## Support Resources

- **LabelImg GitHub**: https://github.com/heartexlabs/labelImg
- **YOLO Format Guide**: https://docs.ultralytics.com/datasets/detect/
- **Roboflow Tutorial**: https://blog.roboflow.com/getting-started-with-roboflow/

For issues with this project:
- Check `results/FULL_DATASET_REPORT.md` for dataset insights
- Review `QUICKSTART_ZED.md` for workflow overview
- Examine sample annotations in `data/labels/` after first few frames

---

**Ready to start?** Launch LabelImg and begin with the first frame!

```bash
cd /Users/chenghsienyu/GitRepos/low_light_yolo
source venv/bin/activate
labelImg data/to_annotate/ data/classes.txt
```
