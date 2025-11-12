# Package Setup Summary

## What Was Done

The `defense_utils` package is now fully set up to support multiple usage methods:

### 1. **Command-Line Tool** (after package installation)

Users can run commands directly in the terminal:

```bash
defense-utils --mode batch_image --input-dir pictures/day2_test --output-dir outputs/day2_test
```

### 2. **Python Module**

Run as a module without installation:

```bash
uv run -m defense_utils --mode batch_image --input-dir pictures/day2_test --output-dir outputs/day2_test
```

### 3. **Direct Python Import**

Import and use functions in Python code:

```python
from defense_utils import combined_processing_and_highlighting, CombinedProcessingConfig
import cv2

image = cv2.imread("image.jpg")
config = CombinedProcessingConfig(threshold_min=30)
result = combined_processing_and_highlighting(image, config)
```

## Files Created/Modified

### Created

- **`defense_utils/__main__.py`** - CLI entry point for the package
- **`PACKAGE_USAGE.md`** - Comprehensive usage guide

### Modified

- **`defense_utils/__init__.py`** - Exports all public functions and classes
- **`pyproject.toml`** - Added script entry point for `defense-utils` command

## Installation Instructions

### For Users (Install as Package)

```bash
# Clone/download the repository
cd delete-bg

# Install as editable package
pip install -e .
# or
uv pip install -e .

# Now use the command
defense-utils --mode batch_image --input-dir pictures/day2_test --output-dir outputs/day2_test
```

### For Development

```bash
# Run directly with uv without installation
uv run main.py --mode batch_image --input-dir pictures/day2_test --output-dir outputs/day2_test

# Or run as module
uv run -m defense_utils --mode batch_image --input-dir pictures/day2_test --output-dir outputs/day2_test
```

## Public API

All functions are now accessible via `defense_utils`:

### Core Functions

- `load_images_from_directory(directory: str)`
- `save_image(image, path: str)`
- `combined_processing_and_highlighting(image, config)`
- `create_video_from_images(images, output_path, config)`
- `extract_image_from_video(video_path, config)`

### Batch Operations

- `create_batch_masked_image(input_dir, output_dir, config, video_config)`
- `create_image_from_extraction_from_video(video_path, save_dir, interval, name_head)`
- `make_video_from_images(input_dir, output_path, video_config)`

### Configuration Classes

- `CombinedProcessingConfig`
- `VideoCreationConfig`
- `FrameExtractionConfig`

## Usage Examples

### Example 1: As Installed Command

```bash
defense-utils --mode batch_image \
  --input-dir pictures/dataset \
  --output-dir outputs/processed \
  --threshold-min 30 \
  --fps 30
```

### Example 2: As Python Module

```bash
python -m defense_utils --mode extract_images \
  --video-path videos/sample.mp4 \
  --extract-output-dir frames/
```

### Example 3: Direct Import

```python
from defense_utils import combined_processing_and_highlighting, CombinedProcessingConfig
import cv2

image = cv2.imread("test.jpg")
config = CombinedProcessingConfig(threshold_min=30, dim_factor=0.5)
steps = combined_processing_and_highlighting(image, config)
result = steps[-1]  # Final result
cv2.imwrite("output.jpg", result)
```

## Documentation

See `PACKAGE_USAGE.md` for:

- Detailed installation instructions
- All available functions and their signatures
- Configuration options
- Complete CLI reference
- Multiple usage examples

## Backward Compatibility

The `main.py` script still works exactly as before:

```bash
uv run main.py --mode batch_image --input-dir pictures/day2_test --output-dir outputs/day2_test
```

Both `main.py` and `defense_utils/__main__.py` provide the same functionality.
