# Quick Start Guide

## Installation

```bash
# Install as editable package
uv pip install -e .
# or
pip install -e .
```

## Three Ways to Use

### 1️⃣ Module Execution (Recommended)

```bash
# Using uv (recommended)
uv run -m defense_utils --mode batch_image --input-dir pictures/day2_test --output-dir outputs/day2_test

# Or with python directly
python -m defense_utils --mode batch_image --input-dir pictures/day2_test --output-dir outputs/day2_test

# Extract frames
uv run -m defense_utils --mode extract_images --video-path videos/video.mp4 --extract-output-dir frames/

# Create video
uv run -m defense_utils --mode concat_images_to_video --concat-input-dir frames/ --video-output-path output.mp4

# Visualize processing pipeline
uv run -m defense_utils --mode custom
```

### 2️⃣ Command-Line Tool (After PATH Configuration)

After installation, you can use `defense-utils` directly if your Python scripts directory is in PATH:

```bash
defense-utils --mode batch_image --input-dir pictures/day2_test --output-dir outputs/day2_test
```

### 3️⃣ Direct Import (For Developers)

```python
from defense_utils import (
    combined_processing_and_highlighting,
    CombinedProcessingConfig,
    create_batch_masked_image,
    VideoCreationConfig,
)
import cv2

# Single image processing
image = cv2.imread("image.jpg")
config = CombinedProcessingConfig(threshold_min=30, dim_factor=0.5)
steps = combined_processing_and_highlighting(image, config)
result = steps[-1]
cv2.imwrite("output.jpg", result)

# Batch processing
config = CombinedProcessingConfig()
video_config = VideoCreationConfig(fps=30, codec='mp4v')
create_batch_masked_image("input_dir", "output_dir", config, video_config)
```

## Common Commands

```bash
# Batch process images with default config
uv run -m defense_utils --mode batch_image --input-dir input/ --output-dir output/

# Extract frames every 15 frames
uv run -m defense_utils --mode extract_images --video-path video.mp4 --extract-output-dir frames/ --frame-extraction-interval 15

# Create video with custom fps
uv run -m defense_utils --mode concat_images_to_video --concat-input-dir frames/ --video-output-path output.mp4 --fps 30

# Tune image processing
uv run -m defense_utils --mode batch_image --input-dir input/ --output-dir output/ \
  --threshold-min 30 \
  --dilation-kernel-size 35 \
  --dim-factor 0.6
```

## Help

```bash
uv run -m defense_utils --help
```

## Documentation

- **Full guide**: `PACKAGE_USAGE.md`
- **Setup details**: `PACKAGE_SETUP.md`
- **Example commands**: `command.txt`

