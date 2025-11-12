"""CLI entry point for defense_utils package."""

import argparse
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np

from defense_utils.image_loader import load_images_from_directory, save_image
from defense_utils.image_processor import combined_processing_and_highlighting
from defense_utils.models.combined_processing_config import CombinedProcessingConfig
from defense_utils.video_creator import create_video_from_images
from defense_utils.video_frame_manipulator import extract_image_from_video
from defense_utils.models.video_creation_config import VideoCreationConfig
from defense_utils.models.frame_extraction_config import FrameExtractionConfig


# Global Configuration Variables
COMBINED_CONFIG = CombinedProcessingConfig(
    threshold_min=25,
    threshold_max=255,
    first_dilation_kernel_size=9,
    first_dilation_iterations=1,
    min_area_ratio=0,
    max_area_ratio=0.1,
    median_filter_kernel_size=5,
    dilation_kernel_size=29,
    dilation_iterations=1,
    erosion_kernel_size=3,
    erosion_iterations=1,
    close_kernel_size=99,
    close_iterations=1,
    highlight_min_area=0,
    highlight_max_area_ratio=0.02,
    third_dilation_kernel_size=9,
    third_dilation_iterations=1,
    dim_factor=0,
)

VIDEO_CONFIG = VideoCreationConfig(fps=25, codec='mp4v')

FRAME_EXTRACTION_INTERVAL = 25
FRAME_EXTRACTION_NAME_HEAD = "extracted_frame"


def create_batch_masked_image(
    input_directory: str,
    output_directory: str,
    config: CombinedProcessingConfig | None = None,
    video_config: VideoCreationConfig | None = None
):
    """Process images in directory and create video using combined processing."""
    if config is None:
        config = COMBINED_CONFIG
    if video_config is None:
        video_config = VIDEO_CONFIG

    images = load_images_from_directory(input_directory)
    result_images: list[np.ndarray[Any, Any]] = []

    output_path_obj = Path(output_directory)
    output_path_obj.mkdir(parents=True, exist_ok=True)

    for filename, image in images.items():
        *_, result = combined_processing_and_highlighting(image, config)

        output_path = output_path_obj / filename
        save_image(result, str(output_path))

        result_images.append(result)

    video_output_path = output_path_obj / "result_video.mp4"
    create_video_from_images(result_images, str(video_output_path), video_config)


def create_image_from_extraction_from_video(
    video_path: str,
    save_directory: str,
    frame_extraction_interval: int | None = None,
    frame_extraction_name_head: str | None = None,
):
    """Extract images from video at intervals."""
    if frame_extraction_interval is None:
        frame_extraction_interval = FRAME_EXTRACTION_INTERVAL
    if frame_extraction_name_head is None:
        frame_extraction_name_head = FRAME_EXTRACTION_NAME_HEAD

    frame_config = FrameExtractionConfig(
        interval_of_extraction=frame_extraction_interval,
        image_name_head=frame_extraction_name_head,
    )
    result = extract_image_from_video(video_path, frame_config)

    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)

    for image_name, image_array in result:
        save_image(image_array, str(save_path / image_name))


def make_video_from_images(
    input_directory: str,
    output_video_path: str,
    video_config: VideoCreationConfig | None = None,
):
    """Create a video from images in a directory."""
    if video_config is None:
        video_config = VIDEO_CONFIG

    images = load_images_from_directory(input_directory)
    create_video_from_images(images, output_video_path, video_config)


def custom_main():
    """Visualize processing pipeline steps."""
    image_path = "pictures/day2_train/img_0437.jpg"
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    print(
        "Processing with combined method (two-stage dilation + contour filtering + highlighting)..."
    )
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
        "12. Third Dilation",
        "13. Final Highlighted Image",
    ]

    # Display all processing steps
    num_images = len(all_steps)
    cols = 6
    rows = (num_images + cols - 1) // cols

    fig = plt.figure(figsize=(24, rows * 4))
    fig.suptitle(
        "Complete Combined Processing Pipeline: Image Processing + Highlighting",
        fontsize=18,
        y=0.995,
    )

    for idx, (img, label) in enumerate(zip(all_steps, labels)):
        plt.subplot(rows, cols, idx + 1)
        if len(img.shape) == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        plt.title(label, fontsize=10, fontweight="bold")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

    # Print configuration used
    print("\n" + "=" * 60)
    print("Combined Configuration Used:")
    print("=" * 60)
    print(f"Image Processing:")
    print(f"  - Threshold Min: {COMBINED_CONFIG.threshold_min}")
    print(f"  - Threshold Max: {COMBINED_CONFIG.threshold_max}")
    print(
        f"  - First Dilation Kernel: {COMBINED_CONFIG.first_dilation_kernel_size}x{COMBINED_CONFIG.first_dilation_kernel_size}"
    )
    print(f"  - First Dilation Iterations: {COMBINED_CONFIG.first_dilation_iterations}")
    print(f"  - Min Area Ratio: {COMBINED_CONFIG.min_area_ratio}")
    print(f"  - Max Area Ratio: {COMBINED_CONFIG.max_area_ratio}")
    print(
        f"  - Median Filter Kernel: {COMBINED_CONFIG.median_filter_kernel_size}x{COMBINED_CONFIG.median_filter_kernel_size}"
    )
    print(
        f"  - Second Dilation Kernel: {COMBINED_CONFIG.dilation_kernel_size}x{COMBINED_CONFIG.dilation_kernel_size}"
    )
    print(
        f"  - Erosion Kernel: {COMBINED_CONFIG.erosion_kernel_size}x{COMBINED_CONFIG.erosion_kernel_size}"
    )
    print(
        f"  - Close Kernel: {COMBINED_CONFIG.close_kernel_size}x{COMBINED_CONFIG.close_kernel_size}"
    )
    print(f"\nObject Highlighting:")
    print(f"  - Highlight Min Area: {COMBINED_CONFIG.highlight_min_area}")
    print(
        f"  - Highlight Max Area Ratio: {COMBINED_CONFIG.highlight_max_area_ratio}"
    )
    print(f"  - Dim Factor: {COMBINED_CONFIG.dim_factor}")
    print(f"\nPost-Highlight Processing:")
    print(
        f"  - Third Dilation Kernel: {COMBINED_CONFIG.third_dilation_kernel_size}x{COMBINED_CONFIG.third_dilation_kernel_size}"
    )
    print(
        f"  - Third Dilation Iterations: {COMBINED_CONFIG.third_dilation_iterations}"
    )
    print("=" * 60)
    print(f"\nTotal steps in pipeline: {len(all_steps)}")
    print(f"Use: *_, result = combined_processing_and_highlighting(image, config)")
    print(f"     to get only the final result (third dilation)")
    print("=" * 60)


def main():
    """Main CLI entry point."""
    arg_parser = argparse.ArgumentParser(description="Process videos and images.")
    arg_parser.add_argument(
        "--mode",
        default=None,
        choices=["batch_image", "extract_images", "concat_images_to_video", "custom"],
        help="Mode of operation: 'batch_image' to process images in a directory, "
        "'extract_images' to extract images from video at intervals, "
        "'concat_images_to_video' to create a video from images in a directory, "
        "'custom' to visualize processing steps.",
    )

    # Arguments for batch_image mode
    arg_parser.add_argument(
        "--input-dir",
        help="Input directory for batch_image mode (e.g., pictures/day2_test)",
    )
    arg_parser.add_argument(
        "--output-dir",
        help="Output directory for batch_image mode (e.g., outputs/day2_test)",
    )

    # Arguments for extract_images mode
    arg_parser.add_argument(
        "--video-path",
        help="Video file path for extract_images mode (e.g., videos/P1_VIDEO_5.mp4)",
    )
    arg_parser.add_argument(
        "--extract-output-dir",
        help="Output directory for extracted images (e.g., pictures/p5)",
    )

    # Arguments for concat_images_to_video mode
    arg_parser.add_argument(
        "--concat-input-dir",
        help="Input directory for concat_images_to_video mode (e.g., pictures/day2_train)",
    )
    arg_parser.add_argument(
        "--video-output-path",
        help="Output video path for concat_images_to_video mode (e.g., outputs/day2_train_video.mp4)",
    )

    # Combined Processing Configuration arguments
    arg_parser.add_argument(
        "--threshold-min", type=int, help="Threshold minimum value"
    )
    arg_parser.add_argument(
        "--threshold-max", type=int, help="Threshold maximum value"
    )
    arg_parser.add_argument(
        "--first-dilation-kernel-size", type=int, help="First dilation kernel size"
    )
    arg_parser.add_argument(
        "--first-dilation-iterations", type=int, help="First dilation iterations"
    )
    arg_parser.add_argument(
        "--min-area-ratio",
        type=float,
        help="Minimum area ratio for contour filtering",
    )
    arg_parser.add_argument(
        "--max-area-ratio",
        type=float,
        help="Maximum area ratio for contour filtering",
    )
    arg_parser.add_argument(
        "--median-filter-kernel-size", type=int, help="Median filter kernel size"
    )
    arg_parser.add_argument(
        "--dilation-kernel-size", type=int, help="Second dilation kernel size"
    )
    arg_parser.add_argument(
        "--dilation-iterations", type=int, help="Second dilation iterations"
    )
    arg_parser.add_argument(
        "--erosion-kernel-size", type=int, help="Erosion kernel size"
    )
    arg_parser.add_argument(
        "--erosion-iterations", type=int, help="Erosion iterations"
    )
    arg_parser.add_argument(
        "--close-kernel-size", type=int, help="Morphological close kernel size"
    )
    arg_parser.add_argument(
        "--close-iterations", type=int, help="Morphological close iterations"
    )
    arg_parser.add_argument(
        "--highlight-min-area", type=int, help="Highlight minimum area"
    )
    arg_parser.add_argument(
        "--highlight-max-area-ratio",
        type=float,
        help="Highlight maximum area ratio",
    )
    arg_parser.add_argument(
        "--third-dilation-kernel-size", type=int, help="Third dilation kernel size"
    )
    arg_parser.add_argument(
        "--third-dilation-iterations", type=int, help="Third dilation iterations"
    )
    arg_parser.add_argument(
        "--dim-factor",
        type=float,
        help="Dim factor for background (< 1.0 dims, > 1.0 brightens)",
    )

    # Video creation configuration arguments
    arg_parser.add_argument("--fps", type=int, help="Frames per second for video creation")
    arg_parser.add_argument("--codec", help="Video codec (e.g., mp4v, XVID)")

    # Frame extraction configuration arguments
    arg_parser.add_argument(
        "--frame-extraction-interval", type=int, help="Frame extraction interval"
    )
    arg_parser.add_argument(
        "--frame-extraction-name-head", help="Prefix for extracted frame names"
    )

    args = arg_parser.parse_args()

    # Build config from parsed arguments, using global defaults for unprovided arguments
    config = CombinedProcessingConfig(
        threshold_min=args.threshold_min
        if args.threshold_min is not None
        else COMBINED_CONFIG.threshold_min,
        threshold_max=args.threshold_max
        if args.threshold_max is not None
        else COMBINED_CONFIG.threshold_max,
        first_dilation_kernel_size=args.first_dilation_kernel_size
        if args.first_dilation_kernel_size is not None
        else COMBINED_CONFIG.first_dilation_kernel_size,
        first_dilation_iterations=args.first_dilation_iterations
        if args.first_dilation_iterations is not None
        else COMBINED_CONFIG.first_dilation_iterations,
        min_area_ratio=args.min_area_ratio
        if args.min_area_ratio is not None
        else COMBINED_CONFIG.min_area_ratio,
        max_area_ratio=args.max_area_ratio
        if args.max_area_ratio is not None
        else COMBINED_CONFIG.max_area_ratio,
        median_filter_kernel_size=args.median_filter_kernel_size
        if args.median_filter_kernel_size is not None
        else COMBINED_CONFIG.median_filter_kernel_size,
        dilation_kernel_size=args.dilation_kernel_size
        if args.dilation_kernel_size is not None
        else COMBINED_CONFIG.dilation_kernel_size,
        dilation_iterations=args.dilation_iterations
        if args.dilation_iterations is not None
        else COMBINED_CONFIG.dilation_iterations,
        erosion_kernel_size=args.erosion_kernel_size
        if args.erosion_kernel_size is not None
        else COMBINED_CONFIG.erosion_kernel_size,
        erosion_iterations=args.erosion_iterations
        if args.erosion_iterations is not None
        else COMBINED_CONFIG.erosion_iterations,
        close_kernel_size=args.close_kernel_size
        if args.close_kernel_size is not None
        else COMBINED_CONFIG.close_kernel_size,
        close_iterations=args.close_iterations
        if args.close_iterations is not None
        else COMBINED_CONFIG.close_iterations,
        highlight_min_area=args.highlight_min_area
        if args.highlight_min_area is not None
        else COMBINED_CONFIG.highlight_min_area,
        highlight_max_area_ratio=args.highlight_max_area_ratio
        if args.highlight_max_area_ratio is not None
        else COMBINED_CONFIG.highlight_max_area_ratio,
        third_dilation_kernel_size=args.third_dilation_kernel_size
        if args.third_dilation_kernel_size is not None
        else COMBINED_CONFIG.third_dilation_kernel_size,
        third_dilation_iterations=args.third_dilation_iterations
        if args.third_dilation_iterations is not None
        else COMBINED_CONFIG.third_dilation_iterations,
        dim_factor=args.dim_factor
        if args.dim_factor is not None
        else COMBINED_CONFIG.dim_factor,
    )

    # Build video config
    video_config = VideoCreationConfig(
        fps=args.fps if args.fps is not None else VIDEO_CONFIG.fps,
        codec=args.codec if args.codec is not None else VIDEO_CONFIG.codec,
    )

    # Update frame extraction settings
    frame_extraction_interval = (
        args.frame_extraction_interval
        if args.frame_extraction_interval is not None
        else FRAME_EXTRACTION_INTERVAL
    )
    frame_extraction_name_head = (
        args.frame_extraction_name_head
        if args.frame_extraction_name_head is not None
        else FRAME_EXTRACTION_NAME_HEAD
    )

    if args.mode == "batch_image":
        if not args.input_dir or not args.output_dir:
            arg_parser.error(
                "--input-dir and --output-dir are required for batch_image mode"
            )
        create_batch_masked_image(args.input_dir, args.output_dir, config, video_config)
    elif args.mode == "extract_images":
        if not args.video_path or not args.extract_output_dir:
            arg_parser.error(
                "--video-path and --extract-output-dir are required for extract_images mode"
            )
        create_image_from_extraction_from_video(
            args.video_path,
            args.extract_output_dir,
            frame_extraction_interval,
            frame_extraction_name_head,
        )
    elif args.mode == "concat_images_to_video":
        if not args.concat_input_dir or not args.video_output_path:
            arg_parser.error(
                "--concat-input-dir and --video-output-path are required for concat_images_to_video mode"
            )
        make_video_from_images(
            args.concat_input_dir, args.video_output_path, video_config
        )
    elif args.mode == "custom" or args.mode is None:
        custom_main()
    else:
        arg_parser.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
