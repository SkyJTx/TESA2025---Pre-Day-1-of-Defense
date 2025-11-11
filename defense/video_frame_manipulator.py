import cv2
from typing import Any
import numpy as np

from defense.models.frame_extraction_config import FrameExtractionConfig


def extract_image_from_video(
    video_path: str,
    config: FrameExtractionConfig | None = None
) -> list[tuple[str, np.ndarray[Any, Any]]]:
    """
    Extract images from a video at specified intervals.

    Args:
        video_path: Path to the video file.
        config: Frame extraction configuration parameters.

    Returns:
        List of tuples containing (image_name, image_array).

    Raises:
        ValueError: If video file cannot be opened.
    """
    if config is None:
        config = FrameExtractionConfig()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {video_path}")

    extracted_images: list[tuple[str, np.ndarray[Any, Any]]] = []
    frame_count = 0
    extracted_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % config.interval_of_extraction == 0:
            image_name = f"{config.image_name_head}_{extracted_count:06d}.png"
            extracted_images.append((image_name, frame))
            extracted_count += 1

        frame_count += 1

    cap.release()

    return extracted_images
