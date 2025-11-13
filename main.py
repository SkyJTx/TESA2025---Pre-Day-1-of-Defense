"""Main entry point - visualize processing pipeline."""

import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any
import numpy as np

from defense_utils.image_loader import load_images_from_directory, save_image
from defense_utils.image_processor import combined_processing_and_highlighting
from defense_utils.models.combined_processing_config import CombinedProcessingConfig
from defense_utils.video_creator import create_video_from_images
from defense_utils.video_frame_manipulator import extract_image_from_video
from defense_utils.models.video_creation_config import VideoCreationConfig
from defense_utils.models.frame_extraction_config import FrameExtractionConfig


# Default configuration from the package
General_Combined_Processing_Config = CombinedProcessingConfig(
    threshold_min=60,
    threshold_max=255,
    first_dilation_kernel_size=1,
    first_dilation_iterations=1,
    min_area_ratio=0,
    max_area_ratio=1,
    median_filter_kernel_size=1,
    dilation_kernel_size=55,
    dilation_iterations=1,
    erosion_kernel_size=9,
    erosion_iterations=1,
    close_kernel_size=5,
    close_iterations=1,
    highlight_min_area=0,
    highlight_max_area_ratio=0.02,
    third_dilation_kernel_size=9,
    third_dilation_iterations=1,
    dim_factor=0,
)

# Merged configuration: command-line args override defaults
GLOBAL_COMBINED_PROCESSING_CONFIG = CombinedProcessingConfig(
    threshold_min=60,                    # Override: 60 (default: 25)
    threshold_max=255,                   # Default: 255
    first_dilation_kernel_size=1,        # Override: 1 (default: 9)
    first_dilation_iterations=1,         # Default: 1
    min_area_ratio=0.00001,                    # Default: 0
    max_area_ratio=0.1,                    # Override: 1 (default: 0.1)
    median_filter_kernel_size=1,         # Override: 1 (default: 5)
    dilation_kernel_size=55,             # Override: 55 (default: 29)
    dilation_iterations=1,               # Default: 1
    erosion_kernel_size=9,               # Override: 9 (default: 3)
    erosion_iterations=1,                # Default: 1
    close_kernel_size=5,                 # Override: 5 (default: 99)
    close_iterations=1,                  # Default: 1
    highlight_min_area=0,                # Default: 0
    highlight_max_area_ratio=0.02,       # Default: 0.02
    third_dilation_kernel_size=9,        # Default: 9
    third_dilation_iterations=1,         # Default: 1
    dim_factor=0,                        # Override: 0 (default: 0)
)



def batch_image(input_directory: str, output_directory: str, config: CombinedProcessingConfig | None = None, video_config: VideoCreationConfig | None = None):
    """Process images in directory and create video using combined processing."""
    if config is None:
        config = GLOBAL_COMBINED_PROCESSING_CONFIG
    if video_config is None:
        video_config = VideoCreationConfig(fps=25, codec='mp4v')

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
    print(f"✓ Processed {len(images)} images")
    print(f"✓ Saved to: {output_directory}")
    print(f"✓ Video created: {video_output_path}")


def extract_images(video_path: str, save_directory: str,
    frame_extraction_config: FrameExtractionConfig | None = None               
    ):
    """Extract images from video at intervals."""
    if frame_extraction_config is None:
        frame_config = FrameExtractionConfig(
            interval_of_extraction=25,
            image_name_head="extracted_frame"
        )
    result = extract_image_from_video(video_path, frame_config)

    save_path = Path(save_directory)
    save_path.mkdir(parents=True, exist_ok=True)

    count = 0
    for image_name, image_array in result:
        save_image(image_array, str(save_path / image_name))
        count += 1
    
    print(f"✓ Extracted {count} frames from: {video_path}")
    print(f"✓ Saved to: {save_directory}")


def concat_images_to_video(input_directory: str, output_video_path: str, video_config: VideoCreationConfig | None = None):
    """Create a video from images in a directory."""
    if video_config is None:
        video_config = VideoCreationConfig(fps=25, codec='mp4v')

    images = load_images_from_directory(input_directory)
    create_video_from_images(images, output_video_path, video_config)
    print(f"✓ Created video from {len(images)} images")
    print(f"✓ Output: {output_video_path}")


"""Simple entry point for defense_utils package - runs visualization demo."""


def main():
    """Visualize processing pipeline steps."""
    # Configuration for visualization
    config = CombinedProcessingConfig(
        threshold_min=60,
        threshold_max=255,
        first_dilation_kernel_size=9,
        first_dilation_iterations=1,
        min_area_ratio=0.00001,
        max_area_ratio=0.1,
        median_filter_kernel_size=1,
        dilation_kernel_size=55,
        dilation_iterations=1,
        erosion_kernel_size=9,
        erosion_iterations=1,
        close_kernel_size=5,
        close_iterations=1,
        highlight_min_area=0,
        highlight_max_area_ratio=0.02,
        third_dilation_kernel_size=9,
        third_dilation_iterations=1,
        dim_factor=0,
    )

    image_path = "pictures/p3new/P3_Frame_000004.png"
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")

    print(
        "Processing with combined method (two-stage dilation + contour filtering + highlighting)..."
    )
    all_steps = combined_processing_and_highlighting(image, config)

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
    print(f"  - Threshold Min: {config.threshold_min}")
    print(f"  - Threshold Max: {config.threshold_max}")
    print(
        f"  - First Dilation Kernel: {config.first_dilation_kernel_size}x{config.first_dilation_kernel_size}"
    )
    print(f"  - First Dilation Iterations: {config.first_dilation_iterations}")
    print(f"  - Min Area Ratio: {config.min_area_ratio}")
    print(f"  - Max Area Ratio: {config.max_area_ratio}")
    print(
        f"  - Median Filter Kernel: {config.median_filter_kernel_size}x{config.median_filter_kernel_size}"
    )
    print(
        f"  - Second Dilation Kernel: {config.dilation_kernel_size}x{config.dilation_kernel_size}"
    )
    print(
        f"  - Erosion Kernel: {config.erosion_kernel_size}x{config.erosion_kernel_size}"
    )
    print(
        f"  - Close Kernel: {config.close_kernel_size}x{config.close_kernel_size}"
    )
    print(f"\nObject Highlighting:")
    print(f"  - Highlight Min Area: {config.highlight_min_area}")
    print(f"  - Highlight Max Area Ratio: {config.highlight_max_area_ratio}")
    print(f"  - Dim Factor: {config.dim_factor}")
    print(f"\nPost-Highlight Processing:")
    print(
        f"  - Third Dilation Kernel: {config.third_dilation_kernel_size}x{config.third_dilation_kernel_size}"
    )
    print(f"  - Third Dilation Iterations: {config.third_dilation_iterations}")
    print("=" * 60)
    print(f"\nTotal steps in pipeline: {len(all_steps)}")
    print(f"Use: *_, result = combined_processing_and_highlighting(image, config)")
    print(f"     to get only the final result (third dilation)")
    print("=" * 60)


if __name__ == "__main__":
    # main()
    batch_image("pictures/p3new", "outputs/p3new")
