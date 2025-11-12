import cv2
from pathlib import Path
from typing import Any
import numpy as np

from defense_utils.models.video_creation_config import VideoCreationConfig


def create_video_from_images(
    images: list[np.ndarray[Any, Any]] | dict[str, np.ndarray[Any, Any]],
    output_path: str,
    config: VideoCreationConfig | None = None
) -> None:
    """
    Create a video from a list or dictionary of images.

    Args:
        images: List of images or dictionary mapping filenames to images.
        output_path: Path where the video will be saved.
        config: Video creation configuration parameters.

    Raises:
        ValueError: If no images provided or images have inconsistent dimensions.
    """
    if config is None:
        config = VideoCreationConfig()
    
    if isinstance(images, dict):
        sorted_filenames = sorted(images.keys())
        image_list = [images[filename] for filename in sorted_filenames]
    else:
        image_list = images

    if not image_list:
        raise ValueError("No images provided to create video")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    first_image = image_list[0]
    height, width = first_image.shape[:2]

    fourcc = int(cv2.VideoWriter.fourcc(*config.codec))
    video_writer = cv2.VideoWriter(
        str(output_file), fourcc, config.fps, (width, height))

    if not video_writer.isOpened():
        raise ValueError(f"Failed to create video writer for: {output_path}")

    for image in image_list:
        if image.shape[:2] != (height, width):
            raise ValueError("All images must have the same dimensions")

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        video_writer.write(image)

    video_writer.release()
