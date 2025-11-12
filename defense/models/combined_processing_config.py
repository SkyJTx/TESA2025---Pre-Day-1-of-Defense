from dataclasses import dataclass
import cv2


@dataclass
class CombinedProcessingConfig:
    """Configuration for combined image processing and highlighting pipeline."""
    # Threshold settings
    threshold_min: int = 100
    threshold_max: int = 255

    # First dilation (expand drone area slightly)
    first_dilation_kernel_size: int = 5
    first_dilation_iterations: int = 1

    # Contour filtering
    min_area_ratio: float = 0.0001  # Minimum area ratio relative to image size
    max_area_ratio: float = 0.5     # Maximum area ratio relative to image size

    # Median filter (remove salt-pepper noise after contour filtering)
    # Kernel size for median filter (must be odd, e.g., 3, 5, 7)
    median_filter_kernel_size: int = 5

    # Second dilation (main expansion)
    dilation_kernel_size: int = 99
    dilation_iterations: int = 1

    # Erosion
    erosion_kernel_size: int = 9
    erosion_iterations: int = 1

    # Morphological closing
    close_kernel_size: int = 9
    close_iterations: int = 1

    # Kernel shape for all operations
    kernel_shape: int = cv2.MORPH_RECT

    # Highlighting parameters
    highlight_min_area: int = 70
    highlight_max_area_ratio: float = 0.001
    dim_factor: float = 0.4  # < 1.0 dims background, > 1.0 brightens, = 1.0 unchanged
