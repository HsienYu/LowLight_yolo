# Quick Start: ZED2i SVO Files → Low-Light Detection

**Your Current Status:** ✅ Setup complete, ready for frame extraction

---

## 📁 What You Have

```
✅ Virtual environment set up
✅ YOLOv8s model downloaded (21.5 MB)
✅ CLAHE preprocessing ready
✅ Detection scripts working
✅ 4 SVO2 files ready (388 MB total)
   └── data/videos/*.svo2
```

---

## 🎯 Next Steps (30 minutes)

### Step 1: Install ZED SDK (15 min)

**Download:**
- Visit: https://www.stereolabs.com/developers/release/
- Select: **ZED SDK for macOS** (ARM64 / Apple Silicon)
- Version: 4.x latest

**Install:**
```bash
# Run the downloaded installer (.dmg or .pkg)
# Follow GUI prompts
# This installs both SDK and Python API (pyzed)
```

**Verify:**
```bash
source venv/bin/activate
python -c "import pyzed.sl as sl; print('✓ ZED SDK:', sl.get_sdk_version())"
```

---

### Step 2: Extract Frames (10 min)

```bash
# Activate environment
source venv/bin/activate

# Extract from all 4 SVO files at once
python scripts/extract_svo_frames.py data/videos/ -o data/images/ --batch -i 30 -m 200

# What this does:
# -i 30     → Every 30th frame (~1 fps at 30fps recording)
# -m 200    → Max 200 frames per video = ~800 total
# --batch   → Process all SVO files automatically
```

**Expected output:**
```
data/images/
├── HD2K_SN34500148_19-08-03/    # ~200 frames (main video)
│   ├── frame_000000_left.jpg
│   ├── frame_000030_left.jpg
│   └── ...
├── HD2K_SN34500148_19-37-06/    # ~50 frames
├── HD2K_SN34500148_19-37-43/    # ~100 frames
└── HD2K_SN34500148_21-32-32/    # ~30 frames
```

---

### Step 3: Test Detection (5 min)

```bash
# Test on first extracted frame with comparison
python scripts/detect.py \
    data/images/HD2K_SN34500148_19-08-03/frame_000030_left.jpg \
    --compare

# Check results
open results/comparison/
```

**What to look for:**
- How many people detected?
- Does CLAHE help? (compare the two images)
- Are people missed or misdetected?

---

## 🔄 Alternative: Manual Frame Extraction

If ZED SDK installation fails, use the GUI tool:

```bash
# After SDK installation, open:
ZED_Depth_Viewer

# Then:
# 1. File → Open SVO
# 2. Select: data/videos/HD2K_SN34500148_19-08-03.svo2
# 3. Tools → Export → Images
# 4. Set frame skip: 30
# 5. Output: data/images/
# 6. Export
```

---

## 📊 What to Expect

### Frame Count Estimates
- **250MB video** → ~3-4 min → ~180-240 frames (at 1fps extraction)
- **43MB video** → ~30-40s → ~30-40 frames
- **73MB video** → ~1 min → ~60 frames
- **22MB video** → ~20s → ~20 frames
- **Total**: ~300-400 frames

### Detection Performance (Initial)
- **Without CLAHE**: 30-40% detection rate (baseline)
- **With CLAHE**: 45-55% detection rate (+15%)
- **After fine-tuning** (your goal): 65-75% detection rate

---

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'pyzed'"
→ ZED SDK not installed. Download from stereolabs.com

### Frames are very dark
→ **This is expected!** Low-light environment. CLAHE will enhance them during detection.
→ Use `--clahe` flag when detecting

### Want both left & right cameras?
```bash
python scripts/extract_svo_frames.py data/videos/ -o data/images/ --batch -i 30 -s both
```

### Need more frames?
```bash
# Extract every 15 frames instead of 30 (2fps instead of 1fps)
python scripts/extract_svo_frames.py data/videos/ -o data/images_more/ --batch -i 15
```

---

## ✅ Checklist

- [ ] ZED SDK installed
- [ ] Verified pyzed import works
- [ ] Extracted frames from SVO files
- [ ] Tested detection on sample frame
- [ ] Reviewed comparison results
- [ ] Identified frames for annotation

---

## 🎓 After Frame Extraction

### Option A: Test More Frames
```bash
# Test multiple frames quickly
for file in data/images/HD2K_*/frame_*0000_left.jpg; do
    python scripts/detect.py "$file" --clahe
done
```

### Option B: Start Annotating
```bash
# Install annotation tool
pip install labelImg

# Start annotating
labelImg data/images/HD2K_SN34500148_19-08-03/ data/labels/ -classes person
```

### Option C: Continue to Training
See main [README.md](README.md) for:
- Data preparation
- Fine-tuning on your data
- Evaluation metrics

---

## 📞 Need Help?

1. **ZED SDK issues**: https://community.stereolabs.com/
2. **YOLO detection issues**: Check test_setup.py results
3. **General workflow**: See [README.md](README.md)

---

**Ready?** Start with Step 1: Install ZED SDK! 🚀
