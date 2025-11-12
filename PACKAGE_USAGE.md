# Defense Utils - Package Usage Guide

## Installation

Install the package locally for development:

```bash
# Using uv
uv pip install -e .

# Or using pip
pip install -e .
```

## Usage Methods

### Method 1: Module Execution (Recommended)

After installing the package, use as a Python module:

```bash
# Using uv
uv run -m defense_utils --mode batch_image --input-dir pictures/day2_test --output-dir outputs/day2_test

# Or with python directly
python -m defense_utils --mode batch_image --input-dir pictures/day2_test --output-dir outputs/day2_test

# Extract frames from video
uv run -m defense_utils --mode extract_images --video-path videos/P1_VIDEO_5.mp4 --extract-output-dir pictures/p5

# Create video from images
uv run -m defense_utils --mode concat_images_to_video --concat-input-dir pictures/day2_train --video-output-path outputs/day2_train_video.mp4

# Visualize processing steps
uv run -m defense_utils --mode custom
```

### Method 2: Command-Line Tool

After installation, if your Python scripts directory is in PATH, you can use:

```bash
defense-utils --mode batch_image --input-dir pictures/day2_test --output-dir outputs/day2_test
```

(If the command is not found, use Method 1 instead)

### Method 3: Direct Import in Python Code

Import and use functions directly in your Python code:

```python
from defense_utils import (
    combined_processing_and_highlighting,
    create_batch_masked_image,
    create_video_from_images,
    extract_image_from_video,
    load_images_from_directory,
    save_image,
    CombinedProcessingConfig,
    VideoCreationConfig,
    FrameExtractionConfig,
)

# Create custom config
config = CombinedProcessingConfig(
    threshold_min=30,
    threshold_max=250,
    dilation_kernel_size=35,
    dim_factor=0.5,
)

# Process a single image
import cv2
image = cv2.imread("path/to/image.jpg")
*_, result = combined_processing_and_highlighting(image, config)
cv2.imwrite("output.jpg", result)

# Batch process images
create_batch_masked_image("input_dir", "output_dir", config)

# Extract frames from video
extract_image_from_video(
    "path/to/video.mp4",
    FrameExtractionConfig(interval_of_extraction=15, image_name_head="frame")
)
```

## Available Functions

### Image Processing

- `load_images_from_directory(directory: str) -> dict`
- `save_image(image: np.ndarray, path: str) -> None`
- `combined_processing_and_highlighting(image: np.ndarray, config: CombinedProcessingConfig) -> list[np.ndarray]`

### Video Operations

- `create_video_from_images(images: list | dict, output_path: str, config: VideoCreationConfig) -> None`
- `extract_image_from_video(video_path: str, config: FrameExtractionConfig) -> generator`

### Batch Operations

- `create_batch_masked_image(input_dir: str, output_dir: str, config: CombinedProcessingConfig, video_config: VideoCreationConfig) -> None`
- `create_image_from_extraction_from_video(video_path: str, save_dir: str, interval: int, name_head: str) -> None`
- `make_video_from_images(input_dir: str, output_path: str, config: VideoCreationConfig) -> None`

## Configuration Classes

### CombinedProcessingConfig

Controls image processing pipeline:

```python
config = CombinedProcessingConfig(
    threshold_min=25,
    threshold_max=255,
    first_dilation_kernel_size=9,
    first_dilation_iterations=1,
    min_area_ratio=0,
    max_area_ratio=0.1,
    median_filter_kernel_size=5,
    dilation_kernel_size=29,
    dilation_iterations=1,
    erosion_kernel_size=3,
    erosion_iterations=1,
    close_kernel_size=99,
    close_iterations=1,
    highlight_min_area=0,
    highlight_max_area_ratio=0.02,
    third_dilation_kernel_size=9,
    third_dilation_iterations=1,
    dim_factor=0,
)
```

### VideoCreationConfig

Controls video output:

```python
video_config = VideoCreationConfig(
    fps=25,
    codec='mp4v'
)
```

### FrameExtractionConfig

Controls frame extraction:

```python
frame_config = FrameExtractionConfig(
    interval_of_extraction=25,
    image_name_head="extracted_frame"
)
```

## Examples

### Example 1: Process Images with Custom Config

```bash
defense-utils --mode batch_image \
  --input-dir pictures/dataset \
  --output-dir outputs/processed \
  --threshold-min 30 \
  --threshold-max 250 \
  --dilation-kernel-size 35 \
  --dim-factor 0.5 \
  --fps 30
```

### Example 2: Extract Frames with Custom Interval

```bash
defense-utils --mode extract_images \
  --video-path videos/sample.mp4 \
  --extract-output-dir frames/ \
  --frame-extraction-interval 15 \
  --frame-extraction-name-head frame
```

### Example 3: Create Video from Images

```bash
defense-utils --mode concat_images_to_video \
  --concat-input-dir frames/ \
  --video-output-path output/video.mp4 \
  --fps 30 \
  --codec XVID
```

### Example 4: Python Script Usage

```python
from defense_utils import (
    combined_processing_and_highlighting,
    CombinedProcessingConfig,
)
import cv2

# Load image
img = cv2.imread("test.jpg")

# Process with custom config
config = CombinedProcessingConfig(threshold_min=30, dim_factor=0.6)
steps = combined_processing_and_highlighting(img, config)

# Get final result (last element)
result = steps[-1]
cv2.imwrite("result.jpg", result)
```

## CLI Options

### Mode Options

- `batch_image` - Process all images in a directory
- `extract_images` - Extract frames from video at intervals
- `concat_images_to_video` - Create video from images
- `custom` - Visualize all processing pipeline steps

### Common Arguments

- `--input-dir` - Input directory path
- `--output-dir` - Output directory path
- `--video-path` - Path to video file
- `--extract-output-dir` - Where to save extracted frames

### Processing Arguments

See CombinedProcessingConfig section for full list of tunable parameters.

## Help

Get detailed help:

```bash
defense-utils --help

# Or from main.py
uv run main.py --help
```
