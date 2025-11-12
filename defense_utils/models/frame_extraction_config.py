from dataclasses import dataclass


@dataclass
class FrameExtractionConfig:
    """Configuration for frame extraction parameters."""
    interval_of_extraction: int = 1
    image_name_head: str = "frame"
