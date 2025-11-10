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

    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    dilated = cv2.dilate(inverted_threshold, dilation_kernel, iterations=1)
    
    erosion_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    eroded = cv2.erode(dilated, erosion_kernel, iterations=1)

    closed_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(eroded, cv2.MORPH_CLOSE,
                              closed_kernel, iterations=1)

    result = cv2.bitwise_not(closed)

    return result


def extract_objects_in_range(
    original_image: np.ndarray[Any, Any],
    binary_image: np.ndarray[Any, Any],
    min_area: int = 70,
    max_area_ratio: float = 0.001
) -> np.ndarray[Any, Any]:
    """
    Extract only objects from original image that are within the specified area range.
    
    Args:
        original_image: Original BGR image as a numpy array.
        binary_image: Binary image (white background, black objects).
        min_area: Minimum area in pixels to keep.
        max_area_ratio: Maximum area as ratio of total image.
    
    Returns:
        Original image with only objects in the specified area range, rest is white.
    """
    inverted_binary = cv2.bitwise_not(binary_image)

    contours, _ = cv2.findContours(
        inverted_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = binary_image.shape[0] * binary_image.shape[1]
    max_area = image_area * max_area_ratio

    filtered_mask = np.zeros_like(binary_image)

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            cv2.drawContours(filtered_mask, [contour], 0, 255, -1)

    result = np.full_like(original_image, 255)
    result[filtered_mask > 0] = original_image[filtered_mask > 0]

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


def detect_and_overlay_bounding_boxes(
    original_image: np.ndarray[Any, Any],
    binary_image: np.ndarray[Any, Any],
    min_area: int = 70,
    max_area_ratio: float = 0.001,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_label: bool = True
) -> np.ndarray[Any, Any]:
    """
    Detect connected components and overlay bounding boxes on the original image.

    Args:
        original_image: Original BGR image as a numpy array.
        binary_image: Binary (thresholded) image as a numpy array.
        min_area: Minimum area in pixels to keep.
        max_area_ratio: Maximum area as ratio of total image.
        color: BGR color for bounding box overlay (default green).
        thickness: Thickness of bounding box lines (default 2).
        show_label: Whether to show area labels on bounding boxes.

    Returns:
        Image with overlaid bounding boxes as a numpy array.
    """
    overlay = original_image.copy()

    inverted_binary = cv2.bitwise_not(binary_image)

    contours, _ = cv2.findContours(
        inverted_binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    image_area = binary_image.shape[0] * binary_image.shape[1]
    max_area = image_area * max_area_ratio

    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)
            
            if show_label:
                label = f"{int(area)}px"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(overlay, (x, y - label_size[1] - 4), (x + label_size[0], y), color, -1)
                cv2.putText(overlay, label, (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

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


def create_video_from_images(
    images: list[np.ndarray[Any, Any]] | dict[str, np.ndarray[Any, Any]],
    output_path: str,
    fps: int = 30,
    codec: str = 'mp4v'
) -> None:
    """
    Create a video from a list or dictionary of images.

    Args:
        images: List of images or dictionary mapping filenames to images.
        output_path: Path where the video will be saved.
        fps: Frames per second for the output video (default: 30).
        codec: Four-character code for video codec (default: 'mp4v').

    Raises:
        ValueError: If no images provided or images have inconsistent dimensions.
    """
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

    fourcc = int(cv2.VideoWriter.fourcc(*codec))
    video_writer = cv2.VideoWriter(
        str(output_file), fourcc, fps, (width, height))

    if not video_writer.isOpened():
        raise ValueError(f"Failed to create video writer for: {output_path}")

    for image in image_list:
        if image.shape[:2] != (height, width):
            raise ValueError("All images must have the same dimensions")

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        video_writer.write(image)

    video_writer.release()


def main():
    input_directory = "pictures"
    output_directory = "outputs"

    images = load_images_from_directory(input_directory)
    result_images: list[np.ndarray[Any, Any]] = []
    bbox_images: list[np.ndarray[Any, Any]] = []

    for filename, image in images.items():
        binary_image = processing_image(image)
        result = extract_objects_in_range(image, binary_image, min_area=400)
        bbox_overlay = detect_and_overlay_bounding_boxes(
            image, binary_image, min_area=400, color=(0, 255, 0), thickness=2
        )
        
        output_path = Path(output_directory) / filename
        bbox_output_path = Path(output_directory) / f"bbox_{filename}"
        save_image(result, str(output_path))
        save_image(bbox_overlay, str(bbox_output_path))
        
        result_images.append(result)
        bbox_images.append(bbox_overlay)
    
    video_output_path = Path(output_directory) / "result_video.mp4"
    bbox_video_output_path = Path(output_directory) / "bbox_video.mp4"
    create_video_from_images(
        result_images, str(video_output_path), fps=1, codec='mp4v')
    create_video_from_images(
        bbox_images, str(bbox_video_output_path), fps=1, codec='mp4v')

def main2():
    input_video_path = "videos/data-2.mp4"
    output_video_path = "outputs/result_video_from_video.mp4"
    bbox_output_video_path = "outputs/bbox_video_from_video.mp4"
    
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {input_video_path}")
    
    frame_images: list[np.ndarray[Any, Any]] = []
    bbox_images: list[np.ndarray[Any, Any]] = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        binary_image = processing_image(frame)
        result = extract_objects_in_range(frame, binary_image, min_area=400)
        bbox_overlay = detect_and_overlay_bounding_boxes(
            frame, binary_image, min_area=400, color=(0, 255, 0), thickness=2
        )
        frame_images.append(result)
        bbox_images.append(bbox_overlay)

    cap.release()

    create_video_from_images(
        frame_images, output_video_path, fps=30, codec='mp4v')
    create_video_from_images(
        bbox_images, bbox_output_video_path, fps=30, codec='mp4v')


if __name__ == "__main__":
    main2()
