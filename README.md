# Defense Utils - Image Processing & Background Removal

A powerful Python package for image processing, object detection, and video manipulation. Designed for drone detection and background removal tasks with advanced morphological operations.

## Features

- 🖼️ **Advanced Image Processing**: Multi-stage dilation, erosion, and morphological operations
- 🎯 **Object Detection & Highlighting**: Contour-based detection with configurable thresholds
- 🎥 **Video Processing**: Extract frames and create videos from image sequences
- ⚙️ **Highly Configurable**: 18+ tunable parameters for fine-grained control
- 🚀 **Multiple Usage Methods**: CLI, Python module, or direct import
- 📦 **Easy Installation**: Simple pip/uv installation

## Installation

```bash
# Using uv (recommended)
uv pip install -e .

# Or using pip
pip install -e .
```

## Quick Start

### CLI Usage

```bash
# Batch process images
uv run -m defense_utils --mode batch_image \
  --input-dir pictures/day2_test \
  --output-dir outputs/day2_test

# Extract frames from video
uv run -m defense_utils --mode extract_images \
  --video-path videos/video.mp4 \
  --extract-output-dir frames/

# Create video from images
uv run -m defense_utils --mode concat_images_to_video \
  --concat-input-dir frames/ \
  --video-output-path output.mp4

# Visualize processing pipeline
uv run -m defense_utils --mode custom
```

### Python Import

```python
from defense_utils import (
    combined_processing_and_highlighting,
    CombinedProcessingConfig,
)
import cv2

# Load and process image
image = cv2.imread("image.jpg")
config = CombinedProcessingConfig(
    threshold_min=30,
    dilation_kernel_size=35,
    dim_factor=0.5
)

# Get processing steps (returns list of all intermediate images)
steps = combined_processing_and_highlighting(image, config)

# Get final result
result = steps[-1]
cv2.imwrite("output.jpg", result)
```

## Usage Methods

### Method 1: Module Execution (Recommended)

Run as a Python module with uv or python:

```bash
# Using uv
uv run -m defense_utils --mode batch_image --input-dir input/ --output-dir output/

# Using python
python -m defense_utils --mode batch_image --input-dir input/ --output-dir output/
```

### Method 2: Command-Line Tool

If Python scripts directory is in your PATH:

```bash
defense-utils --mode batch_image --input-dir input/ --output-dir output/
```

### Method 3: Direct Python Import

Import and use functions in your Python code:

```python
from defense_utils import (
    combined_processing_and_highlighting,
    create_video_from_images,
    load_images_from_directory,
    CombinedProcessingConfig,
    VideoCreationConfig,
)

# Single image processing
import cv2
image = cv2.imread("test.jpg")
config = CombinedProcessingConfig(threshold_min=30)
result = combined_processing_and_highlighting(image, config)[-1]

# Batch processing
from defense_utils import create_batch_masked_image
create_batch_masked_image("input_dir/", "output_dir/", config)
```

### Method 4: Using main.py

The traditional script is still available:

```bash
uv run main.py --mode batch_image --input-dir pictures/day2_test --output-dir outputs/day2_test
```

## Available Modes

| Mode | Description |
|------|-------------|
| `batch_image` | Process all images in a directory and create output video |
| `extract_images` | Extract frames from video at specified intervals |
| `concat_images_to_video` | Create video from image sequence |
| `custom` | Visualize all processing pipeline steps |

## Configuration

### Image Processing Parameters

```python
config = CombinedProcessingConfig(
    # Threshold settings
    threshold_min=25,              # Min threshold for image binarization
    threshold_max=255,             # Max threshold value
    
    # First dilation (initial expansion)
    first_dilation_kernel_size=9,  # Kernel size for first dilation
    first_dilation_iterations=1,   # Number of iterations
    
    # Contour filtering
    min_area_ratio=0,              # Min area ratio for contour filtering
    max_area_ratio=0.1,            # Max area ratio (relative to image size)
    
    # Noise removal
    median_filter_kernel_size=5,   # Kernel size for median filter (must be odd)
    
    # Second dilation (main expansion)
    dilation_kernel_size=29,       # Kernel size for main dilation
    dilation_iterations=1,         # Number of iterations
    
    # Erosion
    erosion_kernel_size=3,         # Kernel size for erosion
    erosion_iterations=1,          # Number of iterations
    
    # Morphological closing
    close_kernel_size=99,          # Kernel size for closing operation
    close_iterations=1,            # Number of iterations
    
    # Object highlighting
    highlight_min_area=0,          # Min area for objects to highlight
    highlight_max_area_ratio=0.02, # Max area ratio for highlighting
    
    # Third dilation (post-highlight)
    third_dilation_kernel_size=9,  # Kernel size for third dilation
    third_dilation_iterations=1,   # Number of iterations
    
    # Background dimming
    dim_factor=0,                  # 0=black, 1=original, >1=brighten
)
```

### Video Settings

```python
video_config = VideoCreationConfig(
    fps=25,        # Frames per second
    codec='mp4v'   # Video codec (mp4v, XVID, etc.)
)
```

### Frame Extraction Settings

```python
frame_config = FrameExtractionConfig(
    interval_of_extraction=25,        # Extract every N frames
    image_name_head="extracted_frame" # Prefix for output files
)
```

## Command Examples

### Basic Operations

```bash
# Process images with default settings
uv run -m defense_utils --mode batch_image \
  --input-dir pictures/day2_test \
  --output-dir outputs/day2_test

# Extract every 15th frame from video
uv run -m defense_utils --mode extract_images \
  --video-path videos/drone.mp4 \
  --extract-output-dir frames/ \
  --frame-extraction-interval 15

# Create 30fps video from images
uv run -m defense_utils --mode concat_images_to_video \
  --concat-input-dir frames/ \
  --video-output-path output.mp4 \
  --fps 30
```

### Custom Configuration

```bash
# Batch process with custom image processing parameters
uv run -m defense_utils --mode batch_image \
  --input-dir pictures/dataset \
  --output-dir outputs/processed \
  --threshold-min 30 \
  --threshold-max 250 \
  --dilation-kernel-size 35 \
  --dim-factor 0.5 \
  --fps 30 \
  --codec XVID

# Extract frames with custom naming
uv run -m defense_utils --mode extract_images \
  --video-path videos/sample.mp4 \
  --extract-output-dir frames/ \
  --frame-extraction-interval 10 \
  --frame-extraction-name-head frame
```

### Full Parameter Example

```bash
uv run -m defense_utils --mode batch_image \
  --input-dir pictures/input \
  --output-dir outputs/result \
  --threshold-min 25 \
  --threshold-max 255 \
  --first-dilation-kernel-size 9 \
  --first-dilation-iterations 1 \
  --min-area-ratio 0 \
  --max-area-ratio 0.1 \
  --median-filter-kernel-size 5 \
  --dilation-kernel-size 29 \
  --dilation-iterations 1 \
  --erosion-kernel-size 3 \
  --erosion-iterations 1 \
  --close-kernel-size 99 \
  --close-iterations 1 \
  --highlight-min-area 0 \
  --highlight-max-area-ratio 0.02 \
  --third-dilation-kernel-size 9 \
  --third-dilation-iterations 1 \
  --dim-factor 0 \
  --fps 25 \
  --codec mp4v
```

## API Reference

### Public Functions

#### Image Loading & Saving

```python
load_images_from_directory(directory: str) -> dict[str, np.ndarray]
save_image(image: np.ndarray, path: str) -> None
```

#### Image Processing

```python
combined_processing_and_highlighting(
    image: np.ndarray,
    config: CombinedProcessingConfig | None = None
) -> list[np.ndarray]
```

Returns list of all processing steps:

1. Grayscale
2. Threshold
3. Inverted Threshold
4. First Dilation
5. Contour Filtered
6. Median Filter (Noise Removal)
7. Second Dilation
8. Erosion
9. Morphological Close
10. Binary Result
11. Object Mask (Highlight)
12. Third Dilation
13. Final Highlighted Image

#### Video Operations

```python
create_video_from_images(
    images: list[np.ndarray] | dict[str, np.ndarray],
    output_path: str,
    config: VideoCreationConfig
) -> None

extract_image_from_video(
    video_path: str,
    config: FrameExtractionConfig
) -> Generator[tuple[str, np.ndarray], None, None]
```

#### Batch Operations

```python
create_batch_masked_image(
    input_directory: str,
    output_directory: str,
    config: CombinedProcessingConfig | None = None,
    video_config: VideoCreationConfig | None = None
) -> None

create_image_from_extraction_from_video(
    video_path: str,
    save_directory: str,
    frame_extraction_interval: int | None = None,
    frame_extraction_name_head: str | None = None
) -> None

make_video_from_images(
    input_directory: str,
    output_video_path: str,
    video_config: VideoCreationConfig | None = None
) -> None
```

## Processing Pipeline

The image processing pipeline consists of multiple stages:

`
Input Image
    ↓
Grayscale Conversion
    ↓
Thresholding
    ↓
Inversion
    ↓
First Dilation (expand objects)
    ↓
Contour Filtering (remove noise)
    ↓
Median Filter (remove salt-pepper noise)
    ↓
Second Dilation (main expansion)
    ↓
Erosion (smooth edges)
    ↓
Morphological Close (fill holes)
    ↓
Binary Result
    ↓
Object Detection & Highlighting
    ↓
Third Dilation (post-processing)
    ↓
Final Result
`

## Project Structure

`
project/
├── defense_utils/           # Main package
│   ├── __init__.py         # Public API exports
│   ├── __main__.py         # CLI entry point
│   ├── image_loader.py     # Image I/O operations
│   ├── image_processor.py  # Core processing logic
│   ├── video_creator.py    # Video creation
│   ├── video_frame_manipulator.py  # Frame extraction
│   └── models/             # Configuration classes
│       ├── combined_processing_config.py
│       ├── video_creation_config.py
│       └── frame_extraction_config.py
├── main.py                 # Alternative entry point
├── pyproject.toml         # Package configuration
├── README.md              # This file
├── QUICK_START.md         # Quick reference
├── PACKAGE_USAGE.md       # Detailed usage guide
└── command.txt            # Command examples
`

## Help & Documentation

### Get Help

```bash
# Show all available options
uv run -m defense_utils --help

# View quick start guide
cat QUICK_START.md

# View detailed usage guide
cat PACKAGE_USAGE.md
```

### Additional Documentation

- **QUICK_START.md** - Quick reference with common commands
- **PACKAGE_USAGE.md** - Comprehensive usage guide with examples
- **PACKAGE_SETUP.md** - Implementation details and setup information
- **command.txt** - Additional command examples

## Requirements

- Python >= 3.12
- opencv-contrib-python >= 4.12.0.88
- matplotlib >= 3.10.7

## Development

```bash
# Clone the repository
git clone https://github.com/SkyJTx/TESA2025---Pre-Day-1-of-Defense.git
cd delete-bg

# Install in editable mode
uv pip install -e .

# Run tests (visualize processing pipeline)
uv run -m defense_utils --mode custom
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Your License Here]

## Author

SkyJTx

## Repository

[Click Here to go to Repo](https://github.com/SkyJTx/TESA2025---Pre-Day-1-of-Defense)
