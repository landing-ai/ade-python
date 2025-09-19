"""Configuration for visualization features."""

from typing import Dict, Tuple, Optional

try:
    import cv2
except ImportError:
    cv2 = None  # type: ignore


class VisualizationConfig:
    """Configuration for visualizing parsing results."""

    def __init__(
        self,
        *,
        thickness: int = 2,
        text_bg_color: Tuple[int, int, int] = (211, 211, 211),  # Light gray in BGR
        text_bg_opacity: float = 0.7,
        padding: int = 2,
        font_scale: float = 0.5,
        font: Optional[int] = None,
        color_map: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ):
        """Initialize visualization configuration.

        Args:
            thickness: Thickness of bounding boxes and text in pixels.
            text_bg_color: Background color for text labels in BGR format.
            text_bg_opacity: Opacity of text background (0.0 to 1.0).
            padding: Padding around text labels in pixels.
            font_scale: Scale factor for font size.
            font: OpenCV font type. Defaults to FONT_HERSHEY_SIMPLEX.
            color_map: Dictionary mapping chunk types to BGR colors.
                      If not provided, uses default color scheme.
        """
        self.thickness = thickness
        self.text_bg_color = text_bg_color
        self.text_bg_opacity = text_bg_opacity
        self.padding = padding
        self.font_scale = font_scale

        # Set default font if cv2 is available and font not specified
        if font is None and cv2 is not None:
            self.font = cv2.FONT_HERSHEY_SIMPLEX
        else:
            self.font = font or 0

        # Default color map for different chunk types (BGR format)
        if color_map is None:
            self.color_map = {
                "text": (255, 0, 0),       # Blue
                "table": (0, 255, 0),       # Green
                "figure": (0, 0, 255),      # Red
                "marginalia": (255, 255, 0), # Cyan
                "formula": (255, 0, 255),    # Magenta
                "code": (0, 255, 255),       # Yellow
                "header": (128, 0, 128),    # Purple
                "footer": (128, 128, 0),    # Teal
                "footnote": (0, 128, 128),  # Olive
                "list": (192, 192, 192),    # Silver
                "title": (64, 64, 64),      # Dark gray
                "subtitle": (96, 96, 96),   # Gray
                "caption": (160, 160, 160), # Light gray
            }
        else:
            self.color_map = color_map

    def get_color(self, chunk_type: str) -> Tuple[int, int, int]:
        """Get color for a given chunk type.

        Args:
            chunk_type: Type of the chunk.

        Returns:
            BGR color tuple. Falls back to blue if type not in color map.
        """
        return self.color_map.get(chunk_type, (255, 0, 0))  # Default to blue