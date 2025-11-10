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


def processing_image(image: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """
    Process an image to remove background and convert to black and white.

    Args:
        image: Input image as a numpy array.

    Returns:
        Processed black and white image as a numpy array.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    threshold = cv2.inRange(gray, *(
        np.array([100]), np.array([255])
    ))

    inverted_threshold = cv2.bitwise_not(threshold)

    # upscaled_width = int(image.shape[1] * 2.5)
    # upscaled_height = int(image.shape[0] * 2.5)
    # upscaled = cv2.resize(inverted_threshold, (upscaled_width, upscaled_height), interpolation=cv2.INTER_LINEAR)

    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilated = cv2.dilate(inverted_threshold, dilation_kernel, iterations=1)

    closed_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, closed_kernel, iterations=1)

    result = cv2.bitwise_not(closed)

    return result


def detect_and_overlay_polygons(
    original_image: np.ndarray[Any, Any],
    binary_image: np.ndarray[Any, Any],
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray[Any, Any]:
    """
    Detect connected components (black areas) and overlay their contours on the original image.

    Args:
        original_image: Original BGR image as a numpy array.
        binary_image: Binary (thresholded) image as a numpy array.
        color: BGR color for contour overlay (default green).
        thickness: Thickness of contour lines (default 2).

    Returns:
        Image with overlaid contours as a numpy array.
    """
    overlay = original_image.copy()

    inverted_binary = cv2.bitwise_not(binary_image)

    contours, _ = cv2.findContours(
        inverted_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = original_image.shape[0] * original_image.shape[1]
    min_area = 70
    max_area = image_area * 0.001

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            cv2.drawContours(overlay, [contour], 0, color, thickness)

    return overlay


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


def main():
    input_directory = "pictures"
    output_directory = "outputs"

    images = load_images_from_directory(input_directory)

    for filename, image in images.items():
        binary_image = processing_image(image)
        # overlay_image = detect_and_overlay_polygons(image, binary_image)
        output_path = Path(output_directory) / filename
        save_image(binary_image, str(output_path))


if __name__ == "__main__":
    main()
