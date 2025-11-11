from dataclasses import dataclass


@dataclass
class VideoCreationConfig:
    """Configuration for video creation parameters."""
    fps: int = 30
    codec: str = 'mp4v'
