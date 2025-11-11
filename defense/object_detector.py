import cv2
from typing import Any
import numpy as np

from defense.models.object_detection_config import ObjectDetectionConfig, OverlayConfig
from defense.models.highlight_config import HighlightConfig


def extract_objects_in_range(
    original_image: np.ndarray[Any, Any],
    binary_image: np.ndarray[Any, Any],
    config: ObjectDetectionConfig | None = None
) -> np.ndarray[Any, Any]:
    """
    Extract only objects from original image that are within the specified area range.

    Args:
        original_image: Original BGR image as a numpy array.
        binary_image: Binary image (white background, black objects).
        config: Object detection configuration parameters.

    Returns:
        Original image with only objects in the specified area range, rest is white.
    """
    if config is None:
        config = ObjectDetectionConfig()

    inverted_binary = cv2.bitwise_not(binary_image)

    contours, _ = cv2.findContours(
        inverted_binary,
        config.retrieval_mode,
        config.approximation_method
    )

    image_area = binary_image.shape[0] * binary_image.shape[1]
    max_area = image_area * config.max_area_ratio

    filtered_mask = np.zeros_like(binary_image)

    for contour in contours:
        area = cv2.contourArea(contour)
        if config.min_area < area < max_area:
            cv2.drawContours(filtered_mask, [contour], 0, 255, -1)

    result = np.full_like(original_image, 255)
    result[filtered_mask > 0] = original_image[filtered_mask > 0]

    return result


def highlight_objects_in_range(
    original_image: np.ndarray[Any, Any],
    binary_image: np.ndarray[Any, Any],
    config: HighlightConfig | None = None
) -> np.ndarray[Any, Any]:
    """
    Dim background (White parts of binary_image) and highlight detected objects.

    Args:
        original_image: Original BGR image as a numpy array.
        binary_image: Binary image (white background, black objects).
        config: Highlight configuration parameters.

    Returns:
        Original image with objects in the specified area range highlighted and background dimmed.
    """
    if config is None:
        config = HighlightConfig()

    inverted_binary = cv2.bitwise_not(binary_image)

    contours, _ = cv2.findContours(
        inverted_binary,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = binary_image.shape[0] * binary_image.shape[1]
    max_area = image_area * config.max_area_ratio

    object_mask = np.zeros_like(binary_image)

    for contour in contours:
        area = cv2.contourArea(contour)
        if config.min_area < area < max_area:
            cv2.drawContours(object_mask, [contour], 0, 255, -1)

    result = original_image.copy().astype(np.float32)

    background_mask = cv2.bitwise_not(object_mask)
    background_mask_3ch = cv2.cvtColor(
        background_mask, cv2.COLOR_GRAY2BGR).astype(np.float32) / 255.0

    # Apply brightness adjustment: dim_factor < 1.0 dims, > 1.0 brightens
    result = result * (1 - background_mask_3ch * (1 - config.dim_factor))

    result = np.clip(result, 0, 255).astype(np.uint8)

    return result


def detect_and_overlay_polygons(
    original_image: np.ndarray[Any, Any],
    binary_image: np.ndarray[Any, Any],
    detection_config: ObjectDetectionConfig | None = None,
    overlay_config: OverlayConfig | None = None
) -> np.ndarray[Any, Any]:
    """
    Detect connected components (black areas) and overlay their contours on the original image.

    Args:
        original_image: Original BGR image as a numpy array.
        binary_image: Binary (thresholded) image as a numpy array.
        detection_config: Object detection configuration parameters.
        overlay_config: Overlay visualization configuration parameters.

    Returns:
        Image with overlaid contours as a numpy array.
    """
    if detection_config is None:
        detection_config = ObjectDetectionConfig()
    if overlay_config is None:
        overlay_config = OverlayConfig()

    overlay = original_image.copy()

    inverted_binary = cv2.bitwise_not(binary_image)

    contours, _ = cv2.findContours(
        inverted_binary,
        detection_config.retrieval_mode,
        detection_config.approximation_method
    )

    image_area = original_image.shape[0] * original_image.shape[1]
    max_area = image_area * detection_config.max_area_ratio

    for contour in contours:
        area = cv2.contourArea(contour)
        if detection_config.min_area < area < max_area:
            cv2.drawContours(
                overlay, [contour], 0, overlay_config.color, overlay_config.thickness)

    return overlay


def detect_and_overlay_bounding_boxes(
    original_image: np.ndarray[Any, Any],
    binary_image: np.ndarray[Any, Any],
    detection_config: ObjectDetectionConfig | None = None,
    overlay_config: OverlayConfig | None = None
) -> np.ndarray[Any, Any]:
    """
    Detect connected components and overlay bounding boxes on the original image.

    Args:
        original_image: Original BGR image as a numpy array.
        binary_image: Binary (thresholded) image as a numpy array.
        detection_config: Object detection configuration parameters.
        overlay_config: Overlay visualization configuration parameters.

    Returns:
        Image with overlaid bounding boxes as a numpy array.
    """
    if detection_config is None:
        detection_config = ObjectDetectionConfig()
    if overlay_config is None:
        overlay_config = OverlayConfig()

    overlay = original_image.copy()

    inverted_binary = cv2.bitwise_not(binary_image)

    contours, _ = cv2.findContours(
        inverted_binary,
        detection_config.retrieval_mode,
        detection_config.approximation_method
    )

    image_area = binary_image.shape[0] * binary_image.shape[1]
    max_area = image_area * detection_config.max_area_ratio

    for contour in contours:
        area = cv2.contourArea(contour)
        if detection_config.min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(overlay, (x, y), (x + w, y + h),
                          overlay_config.color, overlay_config.thickness)

            if overlay_config.show_label:
                label = f"{int(area)}px"
                label_size, _ = cv2.getTextSize(
                    label,
                    overlay_config.font,
                    overlay_config.font_scale,
                    overlay_config.font_thickness
                )
                cv2.rectangle(
                    overlay,
                    (x, y - label_size[1] - 4),
                    (x + label_size[0], y),
                    overlay_config.color,
                    -1
                )
                cv2.putText(
                    overlay,
                    label,
                    (x, y - 2),
                    overlay_config.font,
                    overlay_config.font_scale,
                    overlay_config.label_color,
                    overlay_config.font_thickness
                )

    return overlay
