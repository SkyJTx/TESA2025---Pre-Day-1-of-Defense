from dataclasses import dataclass
import cv2


@dataclass
class ObjectDetectionConfig:
    """Configuration for object detection parameters."""
    min_area: int = 70
    max_area_ratio: float = 0.001
    retrieval_mode: int = cv2.RETR_EXTERNAL
    approximation_method: int = cv2.CHAIN_APPROX_SIMPLE


@dataclass
class OverlayConfig:
    """Configuration for overlay visualization parameters."""
    color: tuple[int, int, int] = (0, 255, 0)
    thickness: int = 2
    show_label: bool = True
    font: int = cv2.FONT_HERSHEY_SIMPLEX
    font_scale: float = 0.5
    font_thickness: int = 1
    label_color: tuple[int, int, int] = (255, 255, 255)
