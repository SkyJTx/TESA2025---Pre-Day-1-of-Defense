import argparse
import cv2
from pathlib import Path
from typing import Any
import numpy as np

from defense.image_loader import load_images_from_directory, save_image
from defense.image_processor import processing_image
from defense.models.image_processing_config import ImageProcessingConfig
from defense.object_detector import (
    detect_and_overlay_bounding_boxes,
    highlight_objects_in_range
)
from defense.video_creator import create_video_from_images
from defense.video_frame_manipulator import extract_image_from_video
from defense.models.object_detection_config import ObjectDetectionConfig, OverlayConfig
from defense.models.video_creation_config import VideoCreationConfig
from defense.models.frame_extraction_config import FrameExtractionConfig
from defense.models.highlight_config import HighlightConfig


# Global Configuration Variables (Merged & Simplified)
# Image Processing Settings (Merged)
PROCESSING_CONFIG = ImageProcessingConfig(
    dilation_kernel_size=33,
    threshold_min=60
)

# Highlight Settings (Merged - replaces ObjectDetectionConfig usage)
HIGHLIGHT_CONFIG = HighlightConfig(min_area=0, max_area_ratio=0.01, dim_factor=0.4)

# Overlay Settings (Bounding Box & Polygon)
OVERLAY_CONFIG = OverlayConfig(color=(0, 255, 0), thickness=2)

# Video Creation Settings (Merged)
VIDEO_CONFIG = VideoCreationConfig(fps=25, codec='mp4v')

# Frame Extraction Settings
FRAME_EXTRACTION_INTERVAL = 25
FRAME_EXTRACTION_NAME_HEAD = "extracted_frame"


def create_batch_masked_image(input_directory: str, output_directory: str):
    """Process images in directory and create video using highlight config."""
    images = load_images_from_directory(input_directory)
    result_images: list[np.ndarray[Any, Any]] = []

    output_path_obj = Path(output_directory)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    for filename, image in images.items():
        binary_image = processing_image(image, PROCESSING_CONFIG)
        result = highlight_objects_in_range(
            image, binary_image, HIGHLIGHT_CONFIG)
        
        output_path = output_path_obj / filename
        save_image(result, str(output_path))
        
        result_images.append(result)
    
    video_output_path = output_path_obj / "result_video.mp4"
    create_video_from_images(
        result_images, str(video_output_path), VIDEO_CONFIG)


def create_bounding_box_overlayed_video(
    input_video_path: str,
    output_video_path: str,
    bbox_output_video_path: str
):
    """Process video and create bounding box overlay using highlight config."""
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video file: {input_video_path}")
    
    frame_images: list[np.ndarray[Any, Any]] = []
    bbox_images: list[np.ndarray[Any, Any]] = []

    # Create minimal ObjectDetectionConfig from HighlightConfig for bounding boxes
    detection_config = ObjectDetectionConfig(
        min_area=HIGHLIGHT_CONFIG.min_area,
        max_area_ratio=HIGHLIGHT_CONFIG.max_area_ratio
    )

    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)
    Path(bbox_output_video_path).parent.mkdir(parents=True, exist_ok=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        binary_image = processing_image(frame)
        result = highlight_objects_in_range(
            frame, binary_image, HIGHLIGHT_CONFIG)
        bbox_overlay = detect_and_overlay_bounding_boxes(
            frame, binary_image, detection_config, OVERLAY_CONFIG
        )
        frame_images.append(result)
        bbox_images.append(bbox_overlay)

    cap.release()

    create_video_from_images(
        frame_images, output_video_path, VIDEO_CONFIG)
    create_video_from_images(
        bbox_images, bbox_output_video_path, VIDEO_CONFIG)


def create_masked_video(input_video_path: str, output_video_path: str):
    """Process video and create masked version using highlight config."""
    cap = cv2.VideoCapture(input_video_path)

    frame_images: list[np.ndarray[Any, Any]] = []

    Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        binary_image = processing_image(frame, PROCESSING_CONFIG)
        result = highlight_objects_in_range(
            frame, binary_image, HIGHLIGHT_CONFIG)
        frame_images.append(result)

    cap.release()

    create_video_from_images(
        frame_images, output_video_path, VIDEO_CONFIG)


def create_image_from_extraction_from_video(
    video_path: str,
    save_directory: str
):
    """Extract images from video at intervals."""
    frame_config = FrameExtractionConfig(
        interval_of_extraction=FRAME_EXTRACTION_INTERVAL,
        image_name_head=FRAME_EXTRACTION_NAME_HEAD
    )
    result = extract_image_from_video(
        video_path,
        frame_config
    )

    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)

    for image_name, image_array in result:
        save_image(image_array, str(save_path / image_name))


def main():
    arg_parser = argparse.ArgumentParser(
        description="Process videos and images.")
    arg_parser.add_argument(
        "--mode",
        choices=["batch_image", "bbox_video",
                 "masked_video", "extract_images"],
        help="Mode of operation: 'batch_image' to process images in a directory, "
             "'bbox_video' to create bounding box overlayed video, "
             "'masked_video' to create masked video from input video, "
             "'extract_images' to extract images from video at intervals.",
    )
    args = arg_parser.parse_args()
    if args.mode == "batch_image":
        create_batch_masked_image("pictures/p5", "outputs/p5")
    elif args.mode == "bbox_video":
        create_bounding_box_overlayed_video(
            "videos/data-2.mp4",
            "outputs/result_video_from_video.mp4",
            "outputs/bbox_video_from_video.mp4"
        )
    elif args.mode == "masked_video":
        in_out_paths = [
            ("videos/P1_VIDEO_1.mp4", "outputs/result_P1_VIDEO_1.mp4"),
            ("videos/P1_VIDEO_2.mp4", "outputs/result_P1_VIDEO_2.mp4"),
            ("videos/P1_VIDEO_3.mp4", "outputs/result_P1_VIDEO_3.mp4"),
            ("videos/P1_VIDEO_4.mp4", "outputs/result_P1_VIDEO_4.mp4"),
            ("videos/P1_VIDEO_5.mp4", "outputs/result_P1_VIDEO_5.mp4"),
        ]
        print("Creating masked videos...")
        for in_path, out_path in in_out_paths:
            create_masked_video(
                in_path,
                out_path
            )
            print(f"Created masked video: {out_path}")
    elif args.mode == "extract_images":
        create_image_from_extraction_from_video(
            "videos/P1_VIDEO_5.mp4",
            "pictures/p5"
        )


if __name__ == "__main__":
    main()
