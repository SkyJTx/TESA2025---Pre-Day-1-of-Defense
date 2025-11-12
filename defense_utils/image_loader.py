import cv2
from pathlib import Path
from typing import Any
import numpy as np


def load_image(image_path: str) -> np.ndarray[Any, Any]:
    """
    Load a single image from the specified file path.

    Args:
        image_path: Path to the image file.

    Returns:
        Image as a numpy array in BGR format.

    Raises:
        FileNotFoundError: If the image file does not exist.
        ValueError: If the image cannot be loaded.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    return image


def load_images_from_directory(directory_path: str) -> dict[str, np.ndarray[Any, Any]]:
    """
    Load all images from a directory.

    Args:
        directory_path: Path to the directory containing images.

    Returns:
        Dictionary mapping image filenames to numpy arrays.
    """
    directory = Path(directory_path)
    if not directory.exists():
        raise FileNotFoundError(f"Directory not found: {directory_path}")

    images: dict[str, np.ndarray[Any, Any]] = {}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}

    for file_path in directory.iterdir():
        if file_path.suffix.lower() in image_extensions:
            try:
                images[file_path.name] = load_image(str(file_path))
            except ValueError:
                continue

    return images


def save_image(image: np.ndarray[Any, Any], output_path: str) -> None:
    """
    Save an image to the specified file path.

    Args:
        image: Image to save as a numpy array.
        output_path: Path where the image will be saved.

    Raises:
        ValueError: If the image cannot be saved.
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(output_file), image)
    if not success:
        raise ValueError(f"Failed to save image: {output_path}")
