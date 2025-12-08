# Annotation Workflow Summary

**Status**: Ready for Phase 2 - Annotation  
**Date**: December 2024

---

## Phase 1 Complete ✅

### Baseline Testing Results
- **Dataset**: 2,371 frames from 4 ZED2i videos
- **Detection rate**: 84.3% (1,998/2,371 frames)
- **Average**: 3.23 people per frame
- **Confidence**: Mean 0.529, 26.9% low-confidence detections
- **Performance**: 28.6 FPS (real-time capable)

### Key Deliverables
✅ Complete baseline testing on all 2,371 frames  
✅ 504 frames strategically selected for annotation  
✅ Annotation guide (ANNOTATION_GUIDE.md)  
✅ Dataset preparation scripts  
✅ Validation scripts  
✅ Comprehensive analysis reports

---

## Phase 2: Annotation (Current Step)

### Quick Start

**1. Setup LabelImg** (one-time, ~5 minutes):
```bash
cd /Users/chenghsienyu/GitRepos/low_light_yolo
source venv/bin/activate
pip install labelImg

# Launch
labelImg data/to_annotate/ data/classes.txt
```

**2. Configure LabelImg**:
- Click "PascalVOC" button → Switch to "YOLO" format
- File → Change Save Dir → Select `data/labels/`
- Ready to annotate!

**3. Annotate** (12-17 hours total):
- Press `W` to create bounding box
- Draw box around each person
- Press `Ctrl+S` to save
- Press `D` for next image
- Repeat for all 504 frames

**4. Validate annotations**:
```bash
python scripts/validate_annotations.py data/to_annotate/ data/labels/
```

**5. Prepare dataset**:
```bash
python scripts/prepare_dataset.py
```

---

## Selected Frames Breakdown

**Total**: 504 frames selected from 2,371

| Category | Count | Purpose |
|----------|-------|---------|
| Max crowd (12 people) | 4 | Test complex scenarios |
| High-density video (19-37-43) | 58 | Challenging detection |
| Main dataset (19-08-03) | 376 | Representative samples |
| Low confidence (<0.40) | ~100 | Target improvements |
| Zero detections | 197 | Catch false negatives |
| Minority videos | 50 | Diversity |

**Distribution by video source**:
- HD2K_SN34500148_19-08-03: 376 frames (74.6%)
- HD2K_SN34500148_19-37-43: 58 frames (11.5%)
- HD2K_SN34500148_19-37-06: 29 frames (5.8%)
- HD2K_SN34500148_21-32-32: 41 frames (8.1%)

**Distribution by people count**:
- 0 people: 197 frames (39.1%)
- 1-2 people: 161 frames (32.0%)
- 3-5 people: 78 frames (15.5%)
- 6+ people: 68 frames (13.5%)

**Location**: `data/to_annotate/`  
**Frame list**: `data/selected_for_annotation.txt`

---

## Files Created

### Scripts
- `scripts/select_frames_for_annotation.py` - Frame selection with stratification
- `scripts/validate_annotations.py` - Annotation validation
- `scripts/prepare_dataset.py` - Train/val/test split
- `scripts/batch_detect.py` - Batch detection testing

### Documentation
- `ANNOTATION_GUIDE.md` - Complete annotation tutorial
- `ANNOTATION_WORKFLOW.md` - This workflow summary
- `results/FULL_DATASET_REPORT.md` - Comprehensive baseline analysis
- `results/BASELINE_REPORT.md` - Initial 19-frame analysis

### Data
- `data/to_annotate/` - 504 selected frames (copied)
- `data/classes.txt` - Class definition (`person`)
- `data/selected_for_annotation.txt` - Frame list
- `results/full_baseline_with_clahe.csv` - Complete detection results

---

## Annotation Guidelines (Quick Reference)

### ✅ DO Annotate
- All visible people (even if small/distant/occluded >30%)
- People in low-light, shadows
- Any pose/angle
- Partially visible people

### ❌ DON'T Annotate
- Reflections, shadows
- People <10 pixels tall
- People >70% occluded
- Mannequins, statues, drawings

### Bounding Box Rules
1. **Tight fit** - No excess padding
2. **Head to feet** - Or visible portions only
3. **Individual boxes** - Separate box per person
4. **When unsure** - Mark it anyway (helps learning)

---

## Expected Timeline

| Task | Time | Sessions |
|------|------|----------|
| LabelImg setup | 15 min | 1 |
| Annotate first 50 | 2-3 hours | 1-2 |
| Annotate next 200 | 5-7 hours | 3-4 |
| Annotate last 254 | 5-7 hours | 3-4 |
| Validation & review | 1 hour | 1 |
| **Total** | **13-18 hours** | **8-12 sessions** |

**Recommended pace**: 50-100 frames per session, 1-2 sessions per day

---

## After Annotation Checklist

Once all 504 frames are annotated:

```bash
# 1. Validate annotations
python scripts/validate_annotations.py data/to_annotate/ data/labels/

# Expected output:
# ✓ Found 504 images
# ✓ Found 504 annotation files
# ✓ All annotations in valid format
# ✓ Total annotated people: ~1,600-2,000

# 2. Prepare dataset (train/val/test split)
python scripts/prepare_dataset.py

# This creates:
# - data/train/ (70% = ~353 frames)
# - data/val/ (20% = ~101 frames)  
# - data/test/ (10% = ~50 frames)
# - data/dataset.yaml (config file)

# 3. Review dataset.yaml
cat data/dataset.yaml

# 4. Ready for training!
# python scripts/train.py (to be created next)
```

---

## Troubleshooting

### LabelImg Issues

**Won't start**:
```bash
pip uninstall labelImg
pip install labelImg
# Or: python -m labelImg
```

**Wrong format (PascalVOC instead of YOLO)**:
- Click "PascalVOC" button to toggle to "YOLO"
- Re-save all annotations

**Can't see images**:
- Check: `data/to_annotate/` has 504 PNG files
- Try absolute path in LabelImg

**Not saving**:
```bash
# Create labels directory manually
mkdir -p data/labels
# Check write permissions
chmod 755 data/labels
```

### Annotation Quality

**Missing labels**:
```bash
# Check which frames are missing
ls data/to_annotate/*.png | wc -l  # Should be 504
ls data/labels/*.txt | wc -l       # Should be 504
```

**Invalid format**:
- Run validation: `python scripts/validate_annotations.py data/to_annotate/ data/labels/ -v`
- Fix errors listed in validation report

---

## Next Steps After Annotation

### Immediate (Phase 2B - Training)
1. ✅ Complete annotation (~504 frames)
2. ✅ Validate annotations
3. ✅ Prepare dataset (train/val/test)
4. ⏳ Create training script (`scripts/train.py`)
5. ⏳ Train fine-tuned model (50 epochs, 2-4 hours)

### Expected Fine-tuning Results
| Metric | Current | Target | Stretch |
|--------|---------|--------|---------|
| Detection rate | 84.3% | 90%+ | 95%+ |
| Low confidence | 26.9% | <15% | <10% |
| Avg confidence | 0.529 | 0.65+ | 0.70+ |
| mAP50 | Unknown | 0.65-0.75 | 0.80+ |

### Long-term (Phase 3 - Deployment)
- Evaluate on test set
- Compare with baseline
- Production deployment
- Real-time monitoring

---

## Quick Commands Reference

```bash
# Launch annotation tool
cd /Users/chenghsienyu/GitRepos/low_light_yolo
source venv/bin/activate
labelImg data/to_annotate/ data/classes.txt

# Check progress
ls data/labels/*.txt | wc -l

# Validate annotations
python scripts/validate_annotations.py data/to_annotate/ data/labels/

# Prepare dataset
python scripts/prepare_dataset.py

# View selected frames list
cat data/selected_for_annotation.txt

# Count people in annotations
find data/labels -name "*.txt" -exec cat {} \; | wc -l
```

---

## Support

- **Full guide**: See `ANNOTATION_GUIDE.md`
- **Baseline analysis**: See `results/FULL_DATASET_REPORT.md`
- **Quick start**: See `QUICKSTART_ZED.md`
- **Project README**: See `README.md`

---

**Ready to start?** Launch LabelImg and annotate your first frame! 🚀

```bash
labelImg data/to_annotate/ data/classes.txt
```
