import argparse
import cv2
from pathlib import Path
from typing import Any
import matplotlib.pyplot as plt
import numpy as np

from defense.image_loader import load_images_from_directory, save_image
from defense.image_processor import combined_processing_and_highlighting
from defense.models.combined_processing_config import CombinedProcessingConfig
from defense.video_creator import create_video_from_images
from defense.video_frame_manipulator import extract_image_from_video
from defense.models.video_creation_config import VideoCreationConfig
from defense.models.frame_extraction_config import FrameExtractionConfig


# Global Configuration Variables (Updated for new processing_image)
# Use the new config defaults which include motion/background and contour filters.
# Tune here if needed (e.g., min/max area, thresholds), otherwise defaults are sensible.

# Combined Processing Configuration
COMBINED_CONFIG = CombinedProcessingConfig(
    threshold_min=25,
    threshold_max=255,
    first_dilation_kernel_size=9,
    first_dilation_iterations=1,
    min_area_ratio=0,
    max_area_ratio=0.020,
    median_filter_kernel_size=5,
    dilation_kernel_size=35,
    dilation_iterations=1,
    erosion_kernel_size=9,
    erosion_iterations=1,
    close_kernel_size=99,
    close_iterations=1,
    highlight_min_area=0,
    highlight_max_area_ratio=0.02,
    dim_factor=0
)

# Video Creation Settings
VIDEO_CONFIG = VideoCreationConfig(fps=25, codec='mp4v')

# Frame Extraction Settings
FRAME_EXTRACTION_INTERVAL = 25
FRAME_EXTRACTION_NAME_HEAD = "extracted_frame"


def create_batch_masked_image(input_directory: str, output_directory: str):
    """Process images in directory and create video using combined processing."""
    images = load_images_from_directory(input_directory)
    result_images: list[np.ndarray[Any, Any]] = []

    output_path_obj = Path(output_directory)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    for filename, image in images.items():
        *_, result = combined_processing_and_highlighting(
            image, COMBINED_CONFIG  # Get final highlighted image
        )
        
        output_path = output_path_obj / filename
        save_image(result, str(output_path))  # type: ignore
        
        result_images.append(result)  # type: ignore
    
    video_output_path = output_path_obj / "result_video.mp4"
    create_video_from_images(
        result_images, str(video_output_path), VIDEO_CONFIG)


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


def make_video_from_images(
    input_directory: str,
    output_video_path: str
):
    """Create a video from images in a directory."""
    images = load_images_from_directory(input_directory)

    create_video_from_images(
        images, output_video_path, VIDEO_CONFIG)


def custom_main():
    image_path = "pictures/day2_train/img_0437.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    # Test combined processing and highlighting
    print("Processing with combined method (two-stage dilation + contour filtering + highlighting)...")
    all_steps = combined_processing_and_highlighting(image, COMBINED_CONFIG)

    # Labels for each step
    labels = [
        "1. Grayscale",
        "2. Threshold",
        "3. Inverted Threshold",
        "4. First Dilation",
        "5. Contour Filtered",
        "6. Median Filter (Noise Removal)",
        "7. Second Dilation",
        "8. Erosion",
        "9. Morphological Close",
        "10. Binary Result",
        "11. Object Mask (Highlight)",
        "12. Final Highlighted Image"
    ]

    # Display all processing steps
    num_images = len(all_steps)
    cols = 6
    rows = (num_images + cols - 1) // cols

    fig = plt.figure(figsize=(24, rows * 4))
    fig.suptitle(
        'Complete Combined Processing Pipeline: Image Processing + Highlighting', fontsize=18, y=0.995)

    for idx, (img, label) in enumerate(zip(all_steps, labels)):
        plt.subplot(rows, cols, idx + 1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        plt.title(label, fontsize=10, fontweight='bold')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    # Print configuration used
    print("\n" + "="*60)
    print("Combined Configuration Used:")
    print("="*60)
    print(f"Image Processing:")
    print(f"  - Threshold Min: {COMBINED_CONFIG.threshold_min}")
    print(f"  - Threshold Max: {COMBINED_CONFIG.threshold_max}")
    print(
        f"  - First Dilation Kernel: {COMBINED_CONFIG.first_dilation_kernel_size}x{COMBINED_CONFIG.first_dilation_kernel_size}")
    print(
        f"  - First Dilation Iterations: {COMBINED_CONFIG.first_dilation_iterations}")
    print(f"  - Min Area Ratio: {COMBINED_CONFIG.min_area_ratio}")
    print(f"  - Max Area Ratio: {COMBINED_CONFIG.max_area_ratio}")
    print(
        f"  - Median Filter Kernel: {COMBINED_CONFIG.median_filter_kernel_size}x{COMBINED_CONFIG.median_filter_kernel_size}")
    print(
        f"  - Second Dilation Kernel: {COMBINED_CONFIG.dilation_kernel_size}x{COMBINED_CONFIG.dilation_kernel_size}")
    print(
        f"  - Erosion Kernel: {COMBINED_CONFIG.erosion_kernel_size}x{COMBINED_CONFIG.erosion_kernel_size}")
    print(
        f"  - Close Kernel: {COMBINED_CONFIG.close_kernel_size}x{COMBINED_CONFIG.close_kernel_size}")
    print(f"\nObject Highlighting:")
    print(f"  - Highlight Min Area: {COMBINED_CONFIG.highlight_min_area}")
    print(
        f"  - Highlight Max Area Ratio: {COMBINED_CONFIG.highlight_max_area_ratio}")
    print(f"  - Dim Factor: {COMBINED_CONFIG.dim_factor}")
    print("="*60)
    print(f"\nTotal steps in pipeline: {len(all_steps)}")
    print(f"Use: *_, result = combined_processing_and_highlighting(image, config)")
    print(f"     to get only the final highlighted image")
    print("="*60)


def main():
    arg_parser = argparse.ArgumentParser(
        description="Process videos and images.")
    arg_parser.add_argument(
        "--mode",
        default=None,
        choices=["batch_image", "extract_images", "concat_images_to_video"],
        help="Mode of operation: 'batch_image' to process images in a directory, "
        "'extract_images' to extract images from video at intervals, "
        "'concat_images_to_video' to create a video from images in a directory."
    )
    args = arg_parser.parse_args()
    if args.mode == "batch_image":
        create_batch_masked_image("pictures/day2_test", "outputs/day2_test")
    elif args.mode == "extract_images":
        create_image_from_extraction_from_video(
            "videos/P1_VIDEO_5.mp4",
            "pictures/p5"
        )
    elif args.mode == "concat_images_to_video":
        make_video_from_images(
            "pictures/day2_train",
            "outputs/day2_train_video.mp4"
        )
    else:
        custom_main()

if __name__ == "__main__":
    main()
