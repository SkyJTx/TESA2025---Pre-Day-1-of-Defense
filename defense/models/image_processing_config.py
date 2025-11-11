from dataclasses import dataclass
import cv2


@dataclass
class ImageProcessingConfig:
    """Configuration for image processing parameters."""
    threshold_min: int = 100
    threshold_max: int = 255
    dilation_kernel_size: int = 13
    dilation_iterations: int = 1
    erosion_kernel_size: int = 9
    erosion_iterations: int = 1
    close_kernel_size: int = 5
    close_iterations: int = 1
    kernel_shape: int = cv2.MORPH_RECT
