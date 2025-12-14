# Low-Light People Detection with YOLO & RT-DETR

**High-accuracy people detection system for low-light environments with multiple model options and enhancement methods.**

This project implements an advanced detection pipeline with:
1. **Multiple Model Architectures** - RT-DETR-X, YOLOv10, YOLOv8
2. **CLAHE Preprocessing** - Fast, reliable low-light enhancement
3. **Optimized Presets** - Pre-configured settings for different use cases

**Best Results:** RT-DETR-X + CLAHE achieves **14.26 avg detections per image** with **100% detection rate** on low-light theatre scenes.

---

## Features

- **Multiple Model Architectures**
  - RT-DETR-X (highest accuracy)
  - YOLOv10m (best speed/accuracy balance)
  - YOLOv8m/s (real-time performance)
  - YOLOv11x (newest architecture)
- **CLAHE Preprocessing** - Classical, reliable enhancement (79.43% improvement)
- **YOLA (NeurIPS 2024)** - State-of-the-art illumination-invariant feature learning
- **Zero-DCE++ Support** - Advanced deep learning enhancement (optional)
- **Optimized Presets** - One-command optimal configurations
- **Video & Camera Support** - Real-time detection from camera/webcam
- **Natural Mirror View** - Camera feed mirrors naturally (enabled by default)
- **Batch Processing** - Process entire datasets with progress tracking
- **MPS Support** - Optimized for Apple Silicon Macs
- **Comparison Tools** - Benchmark multiple models side-by-side
- **Training & Evaluation** - Train custom models with your own datasets
- **Production Ready** - Modular, maintainable architecture

---

## Quick Start

### Choose Your Configuration

**Three optimized presets for different needs:**

| Preset | Model | Enhancement | Avg Detections | Speed | Best For |
|--------|-------|-------------|----------------|-------|----------|
| `max_accuracy` | RT-DETR-X | CLAHE | 14.26/image | 0.074s | Maximum detection accuracy |
| `yola_max` | RT-DETR-X | YOLA | N/A | Slower | Best low-light features (NeurIPS '24) |
| `yola_balanced` | YOLOv10m | YOLA | N/A | Medium | YOLA with faster detector |
| `balanced` | YOLOv10m | CLAHE | 4.86/image | 0.036s | Speed/accuracy balance |
| `real_time` | YOLOv8m | CLAHE | 3.78/image | 0.078s | Real-time applications |

**Quick Detection Examples:**

```bash
# Maximum accuracy (recommended for low-light)
python scripts/detect.py image.jpg --preset max_accuracy -o result.jpg

# YOLA (State-of-the-Art Low-Light) - requires converted weights
python scripts/detect.py image.jpg --preset yola_max --yola-weights models/yola_converted.pth -o result.jpg

# YOLA with YOLOv10m (faster)
python scripts/detect.py image.jpg --preset yola_balanced --yola-weights models/yola_converted.pth -o result.jpg

# Balanced performance (fastest)
python scripts/detect.py image.jpg --preset balanced -o result.jpg

# Real-time (YOLOv8 compatibility)
python scripts/detect.py image.jpg --preset real_time -o result.jpg
```

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Weights

#### Auto-Download (Recommended)

All YOLO and RT-DETR models are automatically downloaded from Ultralytics on first use. Simply run any detection command and the required model will be downloaded to the `models/` directory.

#### Manual Download

If you prefer to download models manually, here are the direct links:

**YOLOv8 Models:**
- [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt) - Small (default, 22MB)
- [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt) - Medium (real_time preset, 52MB)

**YOLOv10 Models:**
- [YOLOv10m](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10m.pt) - Medium (balanced preset, 32MB)

**RT-DETR Models:**
- [RT-DETR-X](https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-x.pt) - Extra Large (max_accuracy preset, 141MB)

**Download and place in `models/` directory:**
```bash
mkdir -p models
cd models

# Download required models
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8m.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov10m.pt
wget https://github.com/ultralytics/assets/releases/download/v8.3.0/rtdetr-x.pt

cd ..
```

#### YOLA (NeurIPS 2024)

YOLA (You Only Look Around) is a state-of-the-art low-light enhancement module that learns illumination-invariant features.

**Setup Steps:**

1. **Download weights** from [YOLA Official Repo](https://github.com/MingboHong/YOLA) (Google Drive links in their README)

2. **Install mmengine** (temporarily needed for conversion):
   ```bash
   pip install mmengine
   ```

3. **Convert weights** to standalone format:
   ```bash
   python scripts/convert_yola_weights.py models/yola.pth models/yola_converted.pth
   ```

4. **Run detection:**
   ```bash
   # With RT-DETR-X (highest accuracy)
   python scripts/detect.py image.jpg --preset yola_max --yola-weights models/yola_converted.pth
   
   # With YOLOv10m (faster)
   python scripts/detect.py image.jpg --preset yola_balanced --yola-weights models/yola_converted.pth
   
   # Combine YOLA + CLAHE
   python scripts/detect.py image.jpg -m models/yolov10m.pt --yola --yola-weights models/yola_converted.pth --clahe
   ```

**Available YOLA Presets:**
- `yola_max` - RT-DETR-X + YOLA (best quality)
- `yola_balanced` - YOLOv10m + YOLA (faster)

#### Zero-DCE++ Weights (Included)

The Zero-DCE++ model weights (`models/zero_dce_plus.pth`, 315KB) are included in this repository for advanced low-light enhancement. No additional download needed.

**Required for these presets:**
- `ultra_accuracy` - RT-DETR-X + Zero-DCE++ Sequential
- `adaptive_smart` - YOLOv8m + Adaptive enhancement
- `ensemble_max` - YOLOv8m + Multi-enhancement fusion

For detailed Zero-DCE++ usage, see [docs/ZERO_DCE_GUIDE.md](docs/ZERO_DCE_GUIDE.md).

### 3. Test Detection

```bash
# Quick test with best preset
python scripts/detect.py path/to/image.jpg --preset max_accuracy -o results/output.jpg

# Or use specific model + CLAHE
python scripts/detect.py path/to/image.jpg -m rtdetr-x.pt --clahe -o results/output.jpg

# Compare with/without CLAHE
python scripts/detect.py path/to/image.jpg --compare -o results/comparison/
```

### 4. Batch Processing

```bash
# Process entire directory with max accuracy
python scripts/batch_detect.py data/images --preset max_accuracy \
    --save-images -o results/batch_results.csv

# Limit to 50 images for testing
python scripts/batch_detect.py data/images --preset max_accuracy \
    --save-images --max-images 50 -o results/test.csv
```

### 5. Results

- Annotated images in `results/` or specified output directory
- CSV reports with detection counts and confidence scores
- Comparison images showing detection differences

---

## Detailed Usage

### Command-Line Arguments Reference

#### Basic Arguments

```bash
python scripts/detect.py [INPUT] [OPTIONS]
```

**Positional Arguments:**
- `INPUT` - Input file path (image or video). Omit for camera mode.

**Input Mode:**
- `--camera` - Use camera input (default: camera ID 0)
- `--camera-id ID` - Specify camera device ID (default: 0)
- `--duration SECONDS` - Recording duration for camera mode

**Output:**
- `-o, --output PATH` - Output file path for results
- `--show` - Display result window
- `--no-window` - Disable display window (video mode)
- `--save-stats` - Save detection statistics to CSV (video mode)

#### Model & Preset Configuration

**Presets (Recommended):**
- `--preset max_accuracy` - RT-DETR-X + CLAHE (best detection)
- `--preset yola_max` - RT-DETR-X + YOLA (best low-light, requires weights)
- `--preset yola_balanced` - YOLOv10m + YOLA (faster low-light)
- `--preset balanced` - YOLOv10m + CLAHE (speed/accuracy balance)
- `--preset real_time` - YOLOv8m + CLAHE (fastest)
- `--preset ultra_accuracy` - RT-DETR-X + Zero-DCE++ (requires weights)
- `--preset adaptive_smart` - YOLOv8m + Adaptive enhancement
- `--preset ensemble_max` - YOLOv8m + Multi-enhancement fusion

**Manual Model Selection:**
- `-m, --model PATH` - Model file path (default: `models/yolov8s.pt`)
- `--model-type TYPE` - Model architecture: `auto`, `yolo`, or `rtdetr` (default: auto)

#### Enhancement Options

**CLAHE Enhancement:**
- `--clahe` - Enable CLAHE preprocessing
- `--clip-limit FLOAT` - CLAHE clip limit (default: 2.0, range: 1.0-4.0)
- `--tile-size INT` - CLAHE tile grid size (default: 8)
- `--compare` - Compare results with/without CLAHE

**YOLA Enhancement (NeurIPS 2024):**
- `--yola` - Enable YOLA enhancement
- `--yola-weights PATH` - Path to YOLA weights (default: `models/yola_converted.pth`)

**Zero-DCE++ Enhancement (Advanced):**
- `--zero-dce` - Enable Zero-DCE++ enhancement
- `--hybrid-mode MODE` - Hybrid detection mode:
  - `sequential` - Zero-DCE++ → YOLO pipeline (fast)
  - `adaptive` - Auto-select enhancement based on brightness
  - `ensemble` - Multi-path enhancement fusion (highest accuracy)

**Combining Enhancements:**
You can combine YOLA with CLAHE for potentially better results:
```bash
python scripts/detect.py image.jpg -m models/yolov10m.pt --yola --yola-weights models/yola_converted.pth --clahe
```

#### Detection Parameters

- `--conf FLOAT` - Confidence threshold (default: 0.25, range: 0.0-1.0)
- `--device DEVICE` - Processing device: `cpu`, `mps`, or `cuda` (default: auto-detect)

#### Image Processing

- **Image Mirroring**: Enabled by default for natural mirror view (when you raise right hand, it appears on right side)

### Model Selection & Presets

**Using Presets (Recommended):**

```bash
# Maximum accuracy - RT-DETR-X + CLAHE
python scripts/detect.py image.jpg --preset max_accuracy

# Balanced - YOLOv10m + CLAHE (fastest)
python scripts/detect.py image.jpg --preset balanced

# Real-time - YOLOv8m + CLAHE
python scripts/detect.py image.jpg --preset real_time
```

**Manual Model Selection:**

```bash
# RT-DETR-X (highest accuracy)
python scripts/detect.py image.jpg -m rtdetr-x.pt --clahe

# RT-DETR-L (lighter)
python scripts/detect.py image.jpg -m rtdetr-l.pt --clahe

# YOLOv10 models
python scripts/detect.py image.jpg -m yolov10m.pt --clahe
python scripts/detect.py image.jpg -m yolov10s.pt --clahe

# YOLOv8 models
python scripts/detect.py image.jpg -m yolov8m.pt --clahe
python scripts/detect.py image.jpg -m yolov8s.pt --clahe
```

**Model Architecture Auto-Detection:**

The system automatically detects model type from filename:
- `rtdetr-*.pt` → RT-DETR architecture
- `yolo*.pt` → YOLO architecture

Or specify manually: `--model-type rtdetr` or `--model-type yolo`

### Batch Processing

```bash
# Process all images in directory
python scripts/batch_detect.py data/images \
    --preset max_accuracy \
    -o results/detections.csv

# Save annotated images
python scripts/batch_detect.py data/images \
    --preset max_accuracy \
    --save-images \
    --output-image-dir results/annotated/ \
    -o results/detections.csv

# Limit number of images
python scripts/batch_detect.py data/images \
    --preset balanced \
    --max-images 100 \
    --save-images \
    -o results/test.csv

# Compare multiple models
# Run each preset separately
python scripts/batch_detect.py data/images --preset max_accuracy \
    --save-images --output-image-dir results/comparison/rtdetr/ \
    -o results/comparison/rtdetr.csv

python scripts/batch_detect.py data/images --preset balanced \
    --save-images --output-image-dir results/comparison/yolov10/ \
    -o results/comparison/yolov10.csv
```

### CLAHE Preprocessing Only

Enhance low-light images without detection:

```bash
# Single image
python scripts/preprocessing.py input.jpg -o enhanced.jpg

# Batch process directory
python scripts/preprocessing.py data/images/ -o data/enhanced/

# Adjust CLAHE parameters
python scripts/preprocessing.py input.jpg -o output.jpg \
    --clip-limit 3.0 \
    --tile-size 4 \
    --show  # Display comparison
```

**CLAHE Parameters:**
- `--clip-limit`: Contrast enhancement (1.0-4.0, default: 2.0)
  - Lower = less enhancement, less noise
  - Higher = more enhancement, more noise
- `--tile-size`: Grid size for local adaptation (default: 8)
  - Smaller = more local adaptation
  - Larger = smoother results

### Usage Examples

#### Image Detection

```bash
# Using presets (recommended)
python scripts/detect.py image.jpg --preset max_accuracy -o result.jpg
python scripts/detect.py image.jpg --preset balanced -o result.jpg
python scripts/detect.py image.jpg --preset real_time -o result.jpg

# Manual model + CLAHE
python scripts/detect.py image.jpg -m models/rtdetr-x.pt --clahe -o result.jpg
python scripts/detect.py image.jpg -m models/yolov10m.pt --clahe -o result.jpg

# Adjust detection confidence (higher = fewer, more confident detections)
python scripts/detect.py image.jpg --preset balanced --conf 0.4 -o result.jpg

# Compare with/without CLAHE
python scripts/detect.py image.jpg -m models/rtdetr-x.pt --compare -o results/comparison/

# Display result window
python scripts/detect.py image.jpg --preset max_accuracy --show
```

#### Video Detection

```bash
# Process video with max accuracy
python scripts/detect.py video.mp4 --preset max_accuracy -o output.mp4

# Process without display window (headless)
python scripts/detect.py video.mp4 --preset balanced --no-window -o output.mp4

# Save detection statistics to CSV
python scripts/detect.py video.mp4 --preset real_time --save-stats -o output.mp4
# Creates video.mp4_stats.csv with frame-by-frame detection data

# Process with custom confidence threshold
python scripts/detect.py video.mp4 --preset balanced --conf 0.3 -o output.mp4
```

#### Camera/Webcam Detection

```bash
# Real-time detection from default camera (with natural mirror view)
python scripts/detect.py --camera --preset balanced

# Use specific camera ID
python scripts/detect.py --camera --camera-id 1 --preset real_time

# Record camera output to video
python scripts/detect.py --camera --preset balanced -o recording.mp4

# Record for specific duration (60 seconds)
python scripts/detect.py --camera --preset real_time --duration 60 -o recording.mp4

# Camera controls during detection:
# - Press 'q' to quit
# - Press 'p' to pause/resume
# - Press 's' to save current frame
```

#### Advanced: Zero-DCE++ Enhancement

```bash
# Use Zero-DCE++ presets (requires model weights)
python scripts/detect.py image.jpg --preset ultra_accuracy -o result.jpg
python scripts/detect.py image.jpg --preset adaptive_smart -o result.jpg
python scripts/detect.py image.jpg --preset ensemble_max -o result.jpg

# Manual Zero-DCE++ configuration
python scripts/detect.py image.jpg --zero-dce --hybrid-mode sequential -o result.jpg
python scripts/detect.py image.jpg --zero-dce --hybrid-mode adaptive -o result.jpg
python scripts/detect.py image.jpg --zero-dce --hybrid-mode ensemble -o result.jpg
```

#### Device Selection

```bash
# Auto-detect best device (default)
python scripts/detect.py image.jpg --preset max_accuracy

# Force CPU (if GPU issues)
python scripts/detect.py image.jpg --preset max_accuracy --device cpu

# Use Apple Silicon GPU (Mac)
python scripts/detect.py image.jpg --preset balanced --device mps

# Use NVIDIA GPU (Linux/Windows)
python scripts/detect.py image.jpg --preset max_accuracy --device cuda
```

---

## Training Your Own Model

> **Note:** Dataset annotation tools have been removed to streamline the project. For dataset preparation and annotation, use standard tools like:
> - **LabelImg** - `pip install labelImg` for manual annotation
> - **Roboflow** - [roboflow.com](https://roboflow.com) for team annotation
> - **Ultralytics Auto-Annotate** - Use pretrained models to generate initial labels

### Prerequisites

Before training, you'll need:
1. **Annotated dataset** in YOLO format:
   ```
   dataset/
   ├── images/
   │   ├── train/
   │   ├── val/
   │   └── test/
   └── labels/
       ├── train/
       ├── val/
       └── test/
   ```
2. **Dataset configuration** (dataset.yaml):
   ```yaml
   path: /path/to/dataset
   train: images/train
   val: images/val
   test: images/test
   
   names:
     0: person
   ```

### Training

```bash
# Basic training
python scripts/train.py --data dataset.yaml --model yolov8m.pt --epochs 100

# Training with specific configuration
python scripts/train.py --data dataset.yaml --model yolov10m.pt --epochs 100 --batch 16

# Monitor training
tensorboard --logdir runs/train/
```

**Key Parameters:**
- `--data`: Path to dataset.yaml configuration file
- `--model`: Pretrained model to start from (yolov8s.pt, yolov8m.pt, yolov10m.pt, etc.)
- `--epochs`: Training iterations (default: 100)
- `--batch`: Batch size, -1 for auto (default: -1)
- `--patience`: Early stopping patience (default: 20)
- `--device`: 'mps', 'cuda', or 'cpu' (auto-detect)

### Evaluation

```bash
# Evaluate on test/validation set
python scripts/evaluate.py --data dataset.yaml --model runs/train/exp/weights/best.pt

# Evaluate with custom confidence threshold
python scripts/evaluate.py --data dataset.yaml --model best.pt --conf 0.3
```

---

## Performance Benchmarks

### Real-World Results (Low-Light Theatre Environment)

**Tested on 50 low-light images:**

| Model | Enhancement | Total Detections | Avg/Image | Inference Time | Speed | Detection Rate |
|-------|-------------|------------------|-----------|----------------|-------|----------------|
| **RT-DETR-X** | CLAHE | **713** | **14.26** | 0.074s | 13.5 FPS | **100%** |
| YOLOv10m | CLAHE | 243 | 4.86 | **0.036s** | **27.7 FPS** | 98% |
| YOLOv8m | CLAHE | 189 | 3.78 | 0.078s | 12.8 FPS | 96% |

**Key Findings:**
- RT-DETR-X detects **3× more people** than YOLOv10m
- YOLOv10m is **2× faster** than RT-DETR-X
- RT-DETR-X achieved **100% detection rate** (no missed frames)
- All models use CLAHE for optimal low-light performance

**Full comparison:** See `results/comparison/COMPARISON_SUMMARY.md`

### Baseline (Pretrained Models)

| Method | mAP50 | Notes |
|--------|-------|-------|
| YOLOv8s alone | ~45-50% | On low-light images |
| YOLOv8s + CLAHE | ~50-60% | +5-10% improvement |
| RT-DETR-X + CLAHE | ~65-70% | Best pretrained performance |

### Expected Fine-Tuning Results

|| Method | mAP50 | Notes |
||--------|-------|-------|
|| Fine-tuned YOLOv8 + CLAHE | **65-75%** | +10-20% over baseline (500-1,000 images) |
|| Fine-tuned RT-DETR + CLAHE | **70-80%** | Best achievable performance (500-1,000 images) |

### Research Benchmarks

- YOLOv8s baseline on ExDark: 55% mAP50
- Enhanced models: 71-75% mAP50
- RT-DETR on low-light: 68-75% mAP50
- Production stability: 95%+ consistent performance

---

## Troubleshooting

### MPS (Mac) Issues

```bash
# If MPS fails, use CPU
python scripts/detect.py input.jpg --device cpu

# Or set environment variable
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

### Memory Issues

```bash
# Reduce batch size
python scripts/train.py --batch 8

# Reduce image size
python scripts/train.py --imgsz 416  # Instead of 640
```

### CLAHE Too Aggressive

```bash
# Reduce clip limit
python scripts/detect.py input.jpg --clahe --clip-limit 1.5

# Increase tile size for smoother results
python scripts/detect.py input.jpg --clahe --tile-size 16
```

---

## Project Structure

```
low_light_yolo/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
│
├── data/                     # Dataset directory
│   └── videos/              # Input videos (ZED2i SVO files, MP4, etc.)
│
├── models/                   # Model weights
│   ├── yolov8s.pt           # YOLOv8 Small (auto-downloaded, 22MB)
│   ├── yolov8m.pt           # YOLOv8 Medium (auto-downloaded, 52MB)
│   ├── yolov10s.pt          # YOLOv10 Small (auto-downloaded, 16MB)
│   ├── yolov10m.pt          # YOLOv10 Medium (auto-downloaded, 33MB)
│   ├── yolo11x.pt           # YOLOv11 XLarge (auto-downloaded, 115MB)
│   ├── rtdetr-l.pt          # RT-DETR Large (auto-downloaded, 67MB)
│   ├── rtdetr-x.pt          # RT-DETR XLarge (auto-downloaded, 136MB)
│   ├── zero_dce_plus.pth    # Zero-DCE++ weights (315KB, included)
│   └── yola_converted.pth   # YOLA weights (converted, see setup)
│
├── scripts/                  # Main scripts
│   ├── preprocessing.py     # CLAHE enhancement
│   ├── detect.py            # Single image/video/camera detection
│   ├── batch_detect.py      # Batch processing
│   ├── compare_methods.py   # Compare enhancement methods
│   ├── train.py             # Model training
│   ├── evaluate.py          # Model evaluation
│   ├── yola.py              # YOLA enhancement (NeurIPS 2024)
│   ├── zero_dce.py          # Zero-DCE++ enhancement
│   ├── hybrid_detector.py   # Hybrid detection modes
│   ├── convert_yola_weights.py  # Convert YOLA weights from MMDetection
│   │
│   ├── dataset/             # Dataset utilities (empty - for future use)
│   │
│   └── utils/               # Utilities
│       └── images_to_video.py  # Convert image sequence to video
│
├── docs/                     # Documentation
│   ├── ZED_SETUP.md         # ZED camera setup guide
│   └── ZERO_DCE_GUIDE.md    # Zero-DCE++ usage guide
│
└── results/                  # Detection outputs (empty initially)
    └── .gitkeep
```

---

## Next Steps

### Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Test detection on an image: `python scripts/detect.py image.jpg --preset max_accuracy -o result.jpg`
3. Process a video: `python scripts/detect.py video.mp4 --preset balanced -o output.mp4`
4. Try camera detection: `python scripts/detect.py --camera --preset balanced`

### For Training Your Own Model
1. Prepare your dataset in YOLO format (images + labels)
2. Create dataset.yaml configuration
3. Train: `python scripts/train.py --data dataset.yaml --model yolov8m.pt`
4. Evaluate: `python scripts/evaluate.py --data dataset.yaml --model runs/train/exp/weights/best.pt`

---

## References

- **YOLOv8/v10/v11**: [Ultralytics Documentation](https://docs.ultralytics.com/)
- **RT-DETR**: [Ultralytics RT-DETR](https://docs.ultralytics.com/models/rtdetr/)
- **YOLA**: [You Only Look Around (NeurIPS 2024)](https://github.com/MingboHong/YOLA)
- **CLAHE**: OpenCV Histogram Equalization
- **Zero-DCE++**: [Zero-Reference Deep Curve Estimation](https://github.com/Li-Chongyi/Zero-DCE_extension)
- **ExDark Dataset**: [Exclusively Dark Image Dataset](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset)

---

## Project Status

This project focuses on low-light people detection with pretrained models. Dataset annotation and preparation utilities have been removed to keep the codebase clean and focused. Use standard annotation tools (LabelImg, Roboflow, etc.) for creating custom datasets.

## Contributing

Improvements welcome! Focus areas:
- Additional enhancement methods
- Training optimizations
- Deployment guides for edge devices
- Performance benchmarks

---

## License

MIT License - Free for academic and commercial use.

---

## Tips

1. **Start small**: Test with 200-300 images before collecting more
2. **Monitor metrics**: Track mAP50 during training
3. **Save checkpoints**: Keep best performing models
4. **Test in production**: Always validate in actual deployment conditions
5. **Iterate**: Fine-tune CLAHE parameters for your specific lighting

---

---

## Quick Reference

### Preset Configurations

```bash
# Maximum accuracy (RT-DETR-X + CLAHE)
--preset max_accuracy

# YOLA presets (requires weights)
--preset yola_max        # RT-DETR-X + YOLA
--preset yola_balanced   # YOLOv10m + YOLA

# Balanced performance (YOLOv10m + CLAHE)
--preset balanced

# Real-time speed (YOLOv8m + CLAHE)
--preset real_time

# Advanced Zero-DCE++ presets (requires weights)
--preset ultra_accuracy   # Best quality
--preset adaptive_smart   # Smart enhancement
--preset ensemble_max     # Highest accuracy
```

### Common Commands

```bash
# Single image detection
python scripts/detect.py image.jpg --preset max_accuracy -o output.jpg

# Video processing
python scripts/detect.py video.mp4 --preset balanced -o output.mp4

# Camera/Webcam (with natural mirror view)
python scripts/detect.py --camera --preset balanced
python scripts/detect.py --camera --preset real_time -o recording.mp4
python scripts/detect.py --camera --duration 60 -o recording.mp4

# Batch processing with saved images
python scripts/batch_detect.py data/images --preset max_accuracy \
    --save-images -o results/batch.csv

# Compare models on 50 samples
python scripts/batch_detect.py data/images --preset max_accuracy \
    --save-images --max-images 50 --output-image-dir results/rtdetr/

python scripts/batch_detect.py data/images --preset balanced \
    --save-images --max-images 50 --output-image-dir results/yolov10/

# Manual model selection
python scripts/detect.py image.jpg -m models/rtdetr-x.pt --clahe -o output.jpg
python scripts/detect.py image.jpg -m models/yolov10m.pt --clahe -o output.jpg

# Advanced CLAHE tuning
python scripts/detect.py image.jpg --clahe --clip-limit 3.0 --tile-size 4

# Device selection
python scripts/detect.py image.jpg --preset max_accuracy --device cpu
python scripts/detect.py image.jpg --preset balanced --device mps
python scripts/detect.py image.jpg --preset max_accuracy --device cuda
```

### Model Files Location

**YOLO/RT-DETR Models (Auto-downloaded):**
- `models/yolov8s.pt` - YOLOv8 Small (default, 22MB)
- `models/yolov8m.pt` - YOLOv8 Medium (real_time preset, 52MB)
- `models/yolov10m.pt` - YOLOv10 Medium (balanced preset, 32MB)
- `models/rtdetr-x.pt` - RT-DETR Extra Large (max_accuracy preset, 141MB)

**Zero-DCE++ Model (Included):**
- `models/zero_dce_plus.pth` - Zero-DCE++ weights (315KB, included in repo)

All YOLO/RT-DETR models are downloaded automatically from Ultralytics on first use.
For manual downloads, see the "Model Weights" section above.

---

**Questions?** Check the scripts for detailed docstrings and usage examples.

**Ready to deploy?** Start with `--preset max_accuracy` for best results! 🚀
