# Dataset Preparation Scripts

Scripts for preparing, annotating, and validating datasets for training.

## Files

### prepare_dataset.py
Prepares and splits dataset into train/val/test sets.

**Usage:**
```bash
python scripts/dataset/prepare_dataset.py \
    --images data/images \
    --labels data/labels \
    --output data/ \
    --split 0.8 0.1 0.1
```

### select_frames_for_annotation.py
Selects diverse frames from video data for annotation.

**Usage:**
```bash
python scripts/dataset/select_frames_for_annotation.py \
    --input data/images/ \
    --output data/selected/ \
    --num-frames 500
```

### validate_annotations.py
Validates YOLO format annotations for correctness.

**Usage:**
```bash
python scripts/dataset/validate_annotations.py \
    --images data/images/ \
    --labels data/labels/
```

### auto_annotate.py
Automatically generates annotations using pretrained models.

**Usage:**
```bash
python scripts/dataset/auto_annotate.py \
    --images data/images/ \
    --output data/labels/ \
    --model yolov8s.pt
```

## Workflow

1. **Extract frames** (if from video): Use `scripts/utils/extract_svo_frames.py`
2. **Select frames**: Use `select_frames_for_annotation.py` to choose diverse samples
3. **Annotate**: Use LabelImg or Roboflow, or `auto_annotate.py` for initial labels
4. **Validate**: Use `validate_annotations.py` to check annotations
5. **Prepare**: Use `prepare_dataset.py` to split into train/val/test sets
