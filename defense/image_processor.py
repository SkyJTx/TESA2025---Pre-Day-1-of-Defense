import cv2
import numpy as np
from typing import Any

from defense.models.combined_processing_config import CombinedProcessingConfig


def combined_processing_and_highlighting(
    image: np.ndarray[Any, Any],
    config: CombinedProcessingConfig | None = None,
) -> list[np.ndarray[Any, Any]]:
    """Combined processing pipeline: two-stage dilation, contour filtering, median filtering, and object highlighting.

    This function performs the complete pipeline:
    1. Image processing with two-stage dilation, contour filtering, and median filtering
    2. Object highlighting with background dimming

    All intermediate steps are returned with the final highlighted image as the last element.

    Args:
        image: Input image as a numpy array.
        config: Combined processing configuration parameters.

    Returns:
        List of intermediate processed images with final highlighted image at the end.
        Use *_, result = combined_processing_and_highlighting(...) to get only the final result.
    """
    if config is None:
        config = CombinedProcessingConfig()

    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Threshold
    threshold = cv2.inRange(
        gray, *(np.array([config.threshold_min]), np.array([config.threshold_max])))

    # Step 3: Invert threshold
    inverted_threshold = cv2.bitwise_not(threshold)

    # Step 4: First dilation - expand drone area slightly
    first_dilation_kernel = cv2.getStructuringElement(
        config.kernel_shape,
        (config.first_dilation_kernel_size, config.first_dilation_kernel_size)
    )
    first_dilated = cv2.dilate(
        inverted_threshold, first_dilation_kernel, iterations=config.first_dilation_iterations)

    # Step 5: Contour filtering based on area ratio
    image_area = image.shape[0] * image.shape[1]
    min_area = image_area * config.min_area_ratio
    max_area = image_area * config.max_area_ratio

    # Find contours
    contours, _ = cv2.findContours(
        first_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Create mask with filtered contours
    contour_filtered = np.zeros_like(first_dilated)
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area <= area <= max_area:
            cv2.drawContours(contour_filtered, [contour], -1, 255, -1)

    # Step 6: Median filter - remove salt-pepper noise (white dots)
    median_filtered = cv2.medianBlur(
        contour_filtered, config.median_filter_kernel_size)

    # Step 7: Second dilation - main expansion
    second_dilation_kernel = cv2.getStructuringElement(
        config.kernel_shape,
        (config.dilation_kernel_size, config.dilation_kernel_size)
    )
    second_dilated = cv2.dilate(
        median_filtered, second_dilation_kernel, iterations=config.dilation_iterations)

    # Step 8: Erosion
    erosion_kernel = cv2.getStructuringElement(
        config.kernel_shape,
        (config.erosion_kernel_size, config.erosion_kernel_size)
    )
    eroded = cv2.erode(second_dilated, erosion_kernel,
                       iterations=config.erosion_iterations)

    # Step 9: Morphological closing
    closed_kernel = cv2.getStructuringElement(
        config.kernel_shape,
        (config.close_kernel_size, config.close_kernel_size)
    )
    closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE,
                              closed_kernel, iterations=config.close_iterations)

    # Step 10: Final inversion (binary result)
    binary_result = cv2.bitwise_not(closed)

    # Step 11: Apply highlighting
    # Invert binary to get objects as white
    inverted_binary = cv2.bitwise_not(binary_result)

    # Find contours for highlighting
    highlight_contours, _ = cv2.findContours(
        inverted_binary,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Filter contours by highlight area thresholds
    highlight_max_area = image_area * config.highlight_max_area_ratio
    object_mask = np.zeros_like(binary_result)

    for contour in highlight_contours:
        area = cv2.contourArea(contour)
        if config.highlight_min_area < area < highlight_max_area:
            cv2.drawContours(object_mask, [contour], 0, 255, -1)

    # Apply dimming to background
    result = image.copy().astype(np.float32)
    background_mask = cv2.bitwise_not(object_mask)
    background_mask_3ch = cv2.cvtColor(
        background_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    # Apply brightness adjustment: dim_factor < 1.0 dims, > 1.0 brightens
    result = result * (1 - background_mask_3ch * (1 - config.dim_factor))
    highlighted_result = np.clip(result, 0, 255).astype(np.uint8)

    return [
        gray,
        threshold,
        inverted_threshold,
        first_dilated,
        contour_filtered,
        median_filtered,
        second_dilated,
        eroded,
        closed,
        binary_result,
        object_mask,
        highlighted_result
    ]
