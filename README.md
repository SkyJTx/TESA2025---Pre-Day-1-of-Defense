# Defense Utils - Image Processing & Background Removal

A powerful Python package for image processing, object detection, and video manipulation. Designed for drone detection and background removal tasks with advanced morphological operations.

## Features

- 🖼️ **Advanced Image Processing**: Multi-stage dilation, erosion, and morphological operations
- 🎯 **Object Detection & Highlighting**: Contour-based detection with configurable thresholds
- 🎥 **Video Processing**: Extract frames and create videos from image sequences
- ⚙️ **Highly Configurable**: 18+ tunable parameters for fine-grained control
- **Easy Installation**: Simple pip/uv installation

## Installation

```bash
# Using uv (recommended)
uv add git+https://github.com/SkyJTx/TESA2025---Pre-Day-1-of-Defense.git

# Or using pip
pip install git+https://github.com/SkyJTx/TESA2025---Pre-Day-1-of-Defense.git
```

## Quick Start

### Python Import Usage

```python
from defense_utils import (
    combined_processing_and_highlighting,
    CombinedProcessingConfig,
    load_images_from_directory,
    save_image,
    create_video_from_images,
    VideoCreationConfig,
)
import cv2

# Single image processing
image = cv2.imread("test.jpg")
config = CombinedProcessingConfig(threshold_min=30, dilation_kernel_size=35)
*_, result = combined_processing_and_highlighting(image, config)
cv2.imwrite("output.jpg", result)

# Batch processing
images = load_images_from_directory("input_dir/")
results = []
for filename, img in images.items():
    *_, processed = combined_processing_and_highlighting(img, config)
    save_image(processed, f"output_dir/{filename}")
    results.append(processed)

# Create video from processed images
video_config = VideoCreationConfig(fps=25, codec='mp4v')
create_video_from_images(results, "output_video.mp4", video_config)
```

### Visualize Pipeline

```bash
# Run visualization demo
uv run python main.py
# or
python -m defense_utils
```

## Available Modes

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
Third Dilation (post-processing)
    ↓
Object Detection & Highlighting
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
├── main.py                 # Visualization demo
├── pyproject.toml         # Package configuration
└── README.md              # This file
`

## Requirements

- Python >= 3.12
- opencv-contrib-python >= 4.12.0.88
- matplotlib >= 3.10.7

## Development

```bash
# Clone the repository
git clone https://github.com/SkyJTx/TESA2025---Pre-Day-1-of-Defense.git
cd delete-bg

# Install in editable mode (for development)
pip install -e .

# Run visualization demo
uv run python main.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Author

SkyJTx

## Repository

[Click Here to go to Repo](https://github.com/SkyJTx/TESA2025---Pre-Day-1-of-Defense)
