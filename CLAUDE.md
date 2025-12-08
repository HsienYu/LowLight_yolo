# CLAUDE.md

**Last Updated**: December 6, 2024

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with this repository. It ensures code quality, consistency, and robustness across all changes.

---

## 📋 Quick Reference

### Project Type
Low-light people detection system using YOLO/RT-DETR with Zero-DCE++ enhancement.

### Tech Stack
- **Framework**: Ultralytics YOLO (YOLOv8, YOLOv10, RT-DETR)
- **Enhancement**: CLAHE (OpenCV) + Zero-DCE++ (PyTorch)
- **Platform**: MacOS (MPS), Linux (CUDA), fallback CPU
- **Python**: 3.8+

### Key Performance Metrics (50 images, low-light theatre)
| Method | Model | Detections | Avg/Image | Speed |
|--------|-------|------------|-----------|-------|
| CLAHE | RT-DETR-X | 713 | 14.26 | 0.101s |
| **Zero-DCE Sequential** | RT-DETR-X | **787** | **15.74** | 0.198s |
| Adaptive | YOLOv8m | 143 | 2.86 | 0.196s |
| Ensemble | YOLOv8m | 166 | 3.32 | 0.334s |

---

## 🏗️ Project Structure

```
low_light_yolo/
├── README.md                    # Main documentation
├── CLAUDE.md                    # This file - AI assistant guidance
├── requirements.txt             # Python dependencies
│
├── docs/                        # Documentation
│   ├── ANNOTATION_GUIDE.md
│   ├── ANNOTATION_WORKFLOW.md
│   ├── AUTO_ANNOTATION_GUIDE.md
│   ├── QUICKSTART_ZED.md
│   ├── ZED_SETUP.md
│   └── ZERO_DCE_GUIDE.md
│
├── models/                      # All model weights (*.pt, *.pth)
│   ├── rtdetr-l.pt             # RT-DETR Large (63MB)
│   ├── rtdetr-x.pt             # RT-DETR Extra-large (129MB)
│   ├── yolov10m.pt             # YOLOv10 Medium (32MB)
│   ├── yolov10s.pt             # YOLOv10 Small (16MB)
│   ├── yolov8m.pt              # YOLOv8 Medium (50MB)
│   ├── yolov8s.pt              # YOLOv8 Small (22MB)
│   └── zero_dce_plus.pth       # Zero-DCE++ weights (315KB)
│
├── scripts/                     # Main codebase
│   ├── detect.py               # ⭐ Main detection script
│   ├── batch_detect.py         # Batch processing
│   ├── preprocessing.py        # CLAHE module
│   ├── zero_dce.py            # Zero-DCE++ module
│   ├── hybrid_detector.py     # Hybrid detection modes
│   ├── compare_methods.py     # Benchmarking tool
│   │
│   ├── utils/                 # Utility scripts
│   │   ├── download_zero_dce_weights.py
│   │   └── extract_svo_frames.py
│   │
│   └── dataset/               # Dataset preparation
│       ├── prepare_dataset.py
│       ├── select_frames_for_annotation.py
│       ├── validate_annotations.py
│       └── auto_annotate.py
│
├── data/                       # Data directory (not in git)
│   ├── images/
│   ├── labels/
│   └── dataset.yaml
│
└── results/                    # Output directory (not in git)
    └── [various subdirectories]
```

---

## 🎯 Core Architecture

### 1. Main Detection Entry Point: `scripts/detect.py`

**PeopleDetector Class**:
- Handles 6 preset configurations
- Supports 3 model types: YOLO, RT-DETR, auto-detect
- Manages 3 enhancement methods: CLAHE, Zero-DCE++, Hybrid modes

**Presets**:
```python
PRESETS = {
    # CLAHE-based (standard)
    'max_accuracy': RT-DETR-X + CLAHE,
    'balanced': YOLOv10m + CLAHE,
    'real_time': YOLOv8m + CLAHE,
    
    # Zero-DCE++ based (advanced)
    'ultra_accuracy': RT-DETR-X + Sequential,
    'adaptive_smart': YOLOv8m + Adaptive,
    'ensemble_max': YOLOv8m + Ensemble
}
```

### 2. Enhancement Modules

**`preprocessing.py`**:
- `CLAHEPreprocessor` class
- `apply_clahe()` function
- Fast, CPU-based enhancement

**`zero_dce.py`**:
- `DCENet` model (PyTorch)
- `ZeroDCEEnhancer` class
- Handles weight loading with `e_conv` → `conv` key mapping
- GPU/MPS acceleration

### 3. Hybrid Detection: `scripts/hybrid_detector.py`

**Three modes**:
1. **SequentialDetector**: Zero-DCE++ → YOLO pipeline
   - Best quality/speed balance
   - 60+ FPS on RTX 3090, ~5 FPS on MPS
   
2. **AdaptiveDetector**: Scene-aware selection
   - Analyzes brightness/contrast/noise
   - Auto-selects enhancement method
   - 40-60 FPS on RTX 3090
   
3. **EnsembleDetector**: Multi-path fusion
   - 4 parallel paths with weighted fusion
   - Highest accuracy
   - 20-25 FPS on RTX 3090

---

## 🔧 Essential Commands

### Setup
```bash
# Initial setup
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download Zero-DCE++ weights
python scripts/utils/download_zero_dce_weights.py
```

### Detection (Using Presets - Recommended)
```bash
# Maximum accuracy (CLAHE-based)
python scripts/detect.py image.jpg --preset max_accuracy -o output.jpg

# Ultra accuracy (Zero-DCE++ based, 10% better)
python scripts/detect.py image.jpg --preset ultra_accuracy -o output.jpg

# Balanced performance
python scripts/detect.py image.jpg --preset balanced -o output.jpg

# Scene-adaptive
python scripts/detect.py image.jpg --preset adaptive_smart -o output.jpg
```

### Batch Processing
```bash
# Process entire directory
python scripts/batch_detect.py data/images/ --preset max_accuracy \
    --save-images -o results/batch.csv

# Limit images and save outputs
python scripts/batch_detect.py data/images/ --preset ultra_accuracy \
    --max-images 50 --save-images \
    --output-image-dir results/annotated/ \
    -o results/detections.csv
```

### Benchmarking
```bash
# Compare all enhancement methods
python scripts/compare_methods.py data/images/ \
    --zero-dce-weights models/zero_dce_plus.pth \
    --max-images 50 -o results/comparison/
```

---

## 📝 Code Quality Standards

### When Modifying Code

1. **Always preserve existing patterns**:
   - Use existing import structure
   - Follow established naming conventions
   - Maintain consistent error handling

2. **Device handling**:
   ```python
   # Always support 'auto' device parameter
   def _setup_device(self, device):
       if device and device != 'auto':
           return device
       # Auto-detect: CUDA > MPS > CPU
       if torch.cuda.is_available():
           return 'cuda'
       elif torch.backends.mps.is_available():
           return 'mps'
       return 'cpu'
   ```

3. **Path handling**:
   - Always use `Path` from pathlib
   - Support both absolute and relative paths
   - Use `Path(__file__).parent` for script-relative paths
   - Model weights default: `models/zero_dce_plus.pth`

4. **Error handling**:
   ```python
   try:
       # Operation
   except Exception as e:
       print(f"Error: {e}")
       # Provide fallback or clear error message
   ```

5. **Progress tracking**:
   - Use `tqdm` for batch operations
   - Provide status updates every few operations
   - Show clear progress indicators

### File Organization Rules

1. **Model weights**: Always in `models/`
2. **Scripts**: Main scripts in `scripts/`, utilities in subdirectories
3. **Documentation**: All `.md` files (except README.md and CLAUDE.md) in `docs/`
4. **No hardcoded paths**: Use parameters with sensible defaults

### Testing Requirements

Before committing changes:
```bash
# Test basic detection
python scripts/detect.py [test_image] --preset max_accuracy -o test_output.jpg

# Test Zero-DCE weights loading
python -c "from scripts.zero_dce import ZeroDCEEnhancer; e = ZeroDCEEnhancer('models/zero_dce_plus.pth'); print('✅ OK')"

# Test hybrid modes
python scripts/detect.py [test_image] --preset ultra_accuracy -o test_zdce.jpg
```

---

## ⚠️ Common Issues & Solutions

### Issue 1: Zero-DCE++ Weight Loading Error
**Symptom**: `Missing key(s) in state_dict` or size mismatch

**Solution**: The official weights use `e_conv` prefix. Our code handles this:
```python
# In zero_dce.py, load_weights() method
if 'e_conv1.weight' in state_dict:
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace('e_conv', 'conv')
        new_state_dict[new_key] = value
    state_dict = new_state_dict
```

### Issue 2: Device 'auto' Not Recognized
**Symptom**: `RuntimeError: Expected one of cpu, cuda... device type at start of device string: auto`

**Solution**: All `_setup_device()` methods must handle 'auto':
```python
if device and device != 'auto':
    return device
```

### Issue 3: Boxes Property is Read-Only
**Symptom**: `property 'xyxy' of 'Boxes' object has no setter`

**Solution**: Create new Boxes objects instead of modifying:
```python
from ultralytics.engine.results import Boxes
boxes_data = torch.zeros((len(final_boxes), 6), device=self.device)
boxes_data[:, :4] = torch.from_numpy(final_boxes).to(self.device)
result.boxes = Boxes(boxes_data, result.orig_shape)
```

### Issue 4: MPS/CUDA Compatibility
**Symptom**: Various PyTorch device errors

**Solution**:
```bash
# Force CPU if issues
python scripts/detect.py image.jpg --device cpu --preset max_accuracy
```

---

## 🔬 Architecture-Specific Notes

### Zero-DCE++ (DCENet)
- **Input**: RGB image [B, 3, H, W], range [0, 1]
- **Output**: Enhanced image + curve parameters
- **Architecture**: 7 conv layers with residual connections
- **Iterations**: 8 enhancement iterations
- **Weights**: Must match official Zero-DCE++ format

### YOLO/RT-DETR Models
- **Person class**: class_id = 0
- **Confidence**: Default 0.25, adjust for low-light
- **Input size**: 640x640 default
- **Output**: Boxes (xyxy, conf, cls)

### Hybrid Detectors
- **Sequential**: Single enhancement path, fastest
- **Adaptive**: Dynamic selection based on scene analysis
- **Ensemble**: 4 parallel paths, uses WBF (Weighted Boxes Fusion)

---

## 📊 Performance Optimization

### For Speed
```bash
# Use CLAHE instead of Zero-DCE++
--preset balanced

# Reduce image size
python scripts/detect.py image.jpg --preset max_accuracy --imgsz 416

# Use smaller model
-m yolov8s.pt
```

### For Accuracy
```bash
# Use Zero-DCE++ with RT-DETR-X
--preset ultra_accuracy

# Lower confidence threshold
--conf 0.20

# Use ensemble mode (slower)
--preset ensemble_max
```

---

## 🎓 Development Workflow

### Adding New Features

1. **Test on single image first**
2. **Add to presets if generally useful**
3. **Update this CLAUDE.md file**
4. **Test batch processing**
5. **Update README.md if user-facing**

### Modifying Enhancement Methods

1. Check `preprocessing.py` for CLAHE changes
2. Check `zero_dce.py` for Zero-DCE++ changes
3. Check `hybrid_detector.py` for hybrid mode changes
4. Ensure backward compatibility
5. Test all 6 presets

### Debugging Checklist

- [ ] Check device compatibility (CPU/MPS/CUDA)
- [ ] Verify model weights in `models/` directory
- [ ] Test with both single image and batch
- [ ] Check memory usage with large batches
- [ ] Verify output paths are created correctly
- [ ] Test all presets work

---

## 📚 Additional Resources

- **Main README**: Project overview and usage
- **ZERO_DCE_GUIDE.md**: Detailed Zero-DCE++ documentation (in `docs/`)
- **ZED_SETUP.md**: ZED2i camera setup (in `docs/`)
- **ANNOTATION_WORKFLOW.md**: Dataset annotation guide (in `docs/`)

---

## 🔄 Change Log

### December 6, 2024
- ✅ Fixed Zero-DCE++ weight loading (e_conv key mapping)
- ✅ Fixed device 'auto' parameter handling
- ✅ Fixed Ensemble detector Boxes read-only issue
- ✅ Organized project structure (models/, docs/, scripts/utils/, scripts/dataset/)
- ✅ Completed batch testing for all 3 Zero-DCE modes
- ✅ Removed test files (test_setup.py, test_zero_dce.py)
- ✅ Sequential: 787 detections (15.74 avg, 100% rate)
- ✅ Adaptive: 143 detections (2.86 avg, 92% rate)
- ✅ Ensemble: 166 detections (3.32 avg, 98% rate)

### Previous Updates
- Implemented hybrid detection modes
- Added Zero-DCE++ support
- Created preset system
- Added RT-DETR model support

---

**Remember**: When making changes, always update this file to reflect new patterns, fixes, and architectural decisions. This ensures consistent code quality across all AI-assisted development sessions.
