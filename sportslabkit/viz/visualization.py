import cv2
import numpy as np

from sportslabkit.types import Color, Point, Rect


def get_color(color: str | tuple[int, int, int] | Color) -> Color:
    if isinstance(color, str):
        return Color.from_name(color)
    elif isinstance(color, tuple):
        return Color(*color)
    elif isinstance(color, Color):
        return color
    else:
        raise TypeError(f"Expected color to be of type str, tuple, or Color, got {type(color)}")

def calculate_marker(anchor: Point, width: int, height: int, margin: int) -> np.ndarray:
    """Calculates the marker contour based on the anchor point, width, height, and margin."""
    x, y = anchor.int_xy_tuple
    return np.array([
        [x - width // 2, y - height - margin],
        [x, y - margin],
        [x + width // 2, y - height - margin]
    ])


def draw_circle(image: np.ndarray, center: Point, radius: int, color: Color, thickness: int = 1, line_type: int = cv2.LINE_AA, shift: int = 0) -> None:
    """Draws a circle on a given image."""
    cv2.circle(image, center.int_xy_tuple, radius, color.bgr_tuple, thickness, line_type, shift)

def draw_marker(image: np.ndarray, anchor: Point, color: Color, marker_width: int = 10, marker_height: int = 20, marker_margin: int = 5, line_type: int = cv2.LINE_AA) -> None:
    """Draws a marker on a given image."""
    marker_contour = calculate_marker(anchor, marker_width, marker_height, marker_margin)
    draw_filled_polygon(image, marker_contour, color)
    draw_polygon(image, marker_contour, Color(0, 0, 0), 2, line_type)

def draw_rect(image: np.ndarray, rect: Rect, color: Color, thickness: int = 2, line_type: int = cv2.LINE_AA, shift: int = 0) -> None:
    """Draws a rectangle on a given image."""
    cv2.rectangle(image, rect.top_left.int_xy_tuple, rect.bottom_right.int_xy_tuple, color.bgr_tuple, thickness, line_type, shift)

def draw_polygon(image: np.ndarray, contour: np.ndarray, color: Color, thickness: int = 2, line_type: int = cv2.LINE_AA) -> None:
    """Draws a polygon on a given image."""
    cv2.drawContours(image, [contour], 0, color.bgr_tuple, thickness, line_type)

def draw_filled_rect(image: np.ndarray, rect: Rect, color: Color) -> None:
    """Draws a filled rectangle on a given image."""
    cv2.rectangle(image, rect.top_left.int_xy_tuple, rect.bottom_right.int_xy_tuple, color.bgr_tuple, -1)

def draw_filled_polygon(image: np.ndarray, contour: np.ndarray, color: Color) -> None:
    """Draws a filled polygon on a given image."""
    cv2.drawContours(image, [contour], 0, color.bgr_tuple, -1)

def draw_text(image: np.ndarray, anchor: Point, text: str, color: Color, font_scale: float = 0.7, thickness: int = 2, line_type: int = cv2.LINE_AA, shift: int = 0) -> None:
    cv2.putText(image, text, anchor.int_xy_tuple, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color.bgr_tuple, thickness, line_type, shift)

def draw_ellipse(image: np.ndarray, rect: Rect, color: Color, thickness: int = 2, line_type: int = cv2.LINE_AA, shift: int = 0) -> None:
    cv2.ellipse(
        image,
        center=rect.bottom_center.int_xy_tuple,
        axes=(int(rect.width), int(0.35 * rect.width)),
        angle=0.0,
        startAngle=-45,
        endAngle=235,
        color=color.bgr_tuple,
        thickness=thickness,
        lineType=line_type,
        shift=shift
    )


def draw_caption(image: np.ndarray, rect: Rect, caption: str, text_color: Color = Color(255, 255, 255), text_thickness: int = 2, rect_color: Color=Color(255, 0,0), line_type: int = cv2.LINE_AA, shift: int = 0) -> None:
    """Draws a caption above a given rectangle on an image."""
    # Calculate optimal font scale to fit the text inside the rectangle
    rect_width = rect.bottom_right.x - rect.top_left.x
    (text_width, text_height), _ = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 1, text_thickness)
    font_scale = min(rect_width / text_width, 1.0)

    # Adjust text thickness based on font scale
    adjusted_text_thickness = max(1, int(text_thickness * font_scale))

    # Calculate position for the filled rectangle and text
    text_position = Point(rect.top_left.x, rect.top_left.y - 5)
    filled_rect_bottom_right = Point(rect.bottom_right.x, text_position.y - int(text_height * font_scale) - 5)

    # Draw filled rectangle
    cv2.rectangle(image, text_position.int_xy_tuple, filled_rect_bottom_right.int_xy_tuple, rect_color.bgr_tuple, -1)

    # Draw text
    text_start = Point(text_position.x, text_position.y - 2)
    cv2.putText(image, caption, text_start.int_xy_tuple, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color.bgr_tuple, adjusted_text_thickness, line_type, shift)

def draw_rect_with_caption(image: np.ndarray, rect: Rect, color: Color, caption: str, thickness: int = 2, text_color: Color = Color(255, 255, 255), text_thickness: int = 2, rect_color:Color = Color(0, 0, 0), line_type: int = cv2.LINE_AA, shift: int = 0) -> None:
    """Draws a rectangle with a caption on a given image."""
    # Draw the main rectangle
    draw_rect(image, rect, color, thickness, line_type, shift)

    # Draw caption
    draw_caption(image, rect, caption, text_color, text_thickness, rect_color, line_type, shift)

def draw_ellipse_with_caption(image: np.ndarray, rect: Rect, color: Color, caption: str, thickness: int = 2, text_color: Color = Color(255, 255, 255), text_thickness: int = 2, line_type: int = cv2.LINE_AA, shift: int = 0) -> None:
    """Draws an ellipse with a caption on a given image."""
    # Draw the main ellipse
    draw_ellipse(image, rect, color, thickness, line_type, shift)

    # Draw caption
    caption_rect = Rect(
        x = rect.bottom_center.x - rect.width / 2,
        y = rect.bottom_center.y,
        width = rect.width,
        height = rect.width / 4
    )
    draw_caption(image, caption_rect, caption, text_color, text_thickness, color, line_type, shift)
