from dataclasses import dataclass


@dataclass
class HighlightConfig:
    """Configuration for object highlighting parameters.

    Attributes:
        dim_factor: Brightness multiplier for background (non-negative real number).
                   < 1.0 dims the background, > 1.0 brightens it, = 1.0 leaves unchanged.
        min_area: Minimum object area threshold in pixels.
        max_area_ratio: Maximum object area as a ratio of image area.
    """
    dim_factor: float = 0.4
    min_area: int = 70
    max_area_ratio: float = 0.001
