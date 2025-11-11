import cv2
from typing import Any
import numpy as np

from defense.models.image_processing_config import ImageProcessingConfig


def processing_image(
    image: np.ndarray[Any, Any],
    config: ImageProcessingConfig | None = None
) -> np.ndarray[Any, Any]:
    """
    Process an image to remove background and convert to black and white.

    Args:
        image: Input image as a numpy array.
        config: Image processing configuration parameters.

    Returns:
        Processed black and white image as a numpy array.
    """
    if config is None:
        config = ImageProcessingConfig()
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold = cv2.inRange(gray, *(
        np.array([config.threshold_min]), np.array([config.threshold_max])
    ))

    inverted_threshold = cv2.bitwise_not(threshold)

    dilation_kernel = cv2.getStructuringElement(
        config.kernel_shape,
        (config.dilation_kernel_size, config.dilation_kernel_size)
    )
    dilated = cv2.dilate(
        inverted_threshold,
        dilation_kernel,
        iterations=config.dilation_iterations
    )

    erosion_kernel = cv2.getStructuringElement(
        config.kernel_shape,
        (config.erosion_kernel_size, config.erosion_kernel_size)
    )
    eroded = cv2.erode(
        dilated,
        erosion_kernel,
        iterations=config.erosion_iterations
    )

    closed_kernel = cv2.getStructuringElement(
        config.kernel_shape,
        (config.close_kernel_size, config.close_kernel_size)
    )
    closed = cv2.morphologyEx(
        eroded,
        cv2.MORPH_CLOSE,
        closed_kernel,
        iterations=config.close_iterations
    )

    result = cv2.bitwise_not(closed)

    return result
