# ZED 2i Camera Support - Changelog

## Summary
Added comprehensive ZED 2i stereo camera support to `scripts/detect.py` for real-time people detection.

## Changes Made

### 1. Import Section (Lines 37-42)
- Added optional `pyzed.sl` import with try-except error handling
- Created `ZED_AVAILABLE` flag for feature detection
- Graceful degradation if ZED SDK not installed

### 2. New Method: `detect_camera_zed()` (Lines 819-977)
- Full-featured ZED camera detection method mirroring `detect_camera()` functionality
- **Configuration:**
  - Resolution: HD720 (1280x720)
  - FPS: 30
  - View: LEFT camera
  - Depth mode: NONE (not needed for detection)
  
- **Features:**
  - Real-time people detection with all enhancement methods (CLAHE, Zero-DCE++, YOLA)
  - Video recording to MP4
  - Frame saving (saves to `results/zed_frame_XXXXXX.jpg`)
  - Duration limiting
  - Interactive controls (q/p/s keys)
  - Statistics tracking and summary

- **Camera Info Display:**
  - Camera model
  - Serial number
  - Resolution and FPS

- **Error Handling:**
  - SDK availability check
  - Camera initialization error reporting
  - Frame grab failure detection

### 3. CLI Arguments (Line 1011-1015)
- Added `--zed` flag: enables ZED camera mode
- Works in combination with existing `--camera` flag
- Compatible with all other options (--clahe, --yola, --preset, etc.)

### 4. Main Routing Logic (Lines 1140-1151)
- Enhanced camera mode routing to check `--zed` flag
- Routes to `detect_camera_zed()` when ZED mode enabled
- Validates ZED SDK availability before attempting to use camera
- Provides helpful error messages with documentation reference

## File Statistics
- **Lines Added:** 191
- **Lines Modified:** 8
- **Total Changes:** 199 lines

## Usage Examples

```bash
# Basic ZED camera detection
python scripts/detect.py --camera --zed

# With video recording
python scripts/detect.py --camera --zed -o output/recording.mp4

# With enhanced detection
python scripts/detect.py --camera --zed --clahe --preset max_accuracy

# With time limit
python scripts/detect.py --camera --zed --duration 30
```

## Testing
- ✓ Python syntax validation passed
- ✓ Method signature verification passed
- ✓ CLI help output includes `--zed` flag
- ✓ ZED SDK import works correctly (ZED_AVAILABLE=True)
- ✓ PeopleDetector.detect_camera_zed() method exists and is callable

## Documentation
Created `ZED_CAMERA_USAGE.md` with comprehensive usage guide including:
- Setup requirements
- Usage examples
- Feature list
- Keyboard controls
- Implementation details
- Example output

## Compatibility
- Maintains backward compatibility with existing webcam functionality
- No breaking changes to existing API or CLI arguments
- Works with all existing enhancement methods and presets
- Follows existing code patterns and conventions
