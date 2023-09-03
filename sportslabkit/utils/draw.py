from typing import List, Optional, Tuple, Union

import matplotlib
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

from sportslabkit.logger import logger


def _generate_color_palette(n: int) -> np.ndarray:
    """
    Generates random colors for bounding boxes.
    Args:
        n (int): Number of colors to generate.
    Returns:
        np.ndarray: Array of shape (N, 3) containing the colors in RGB format.
    """
    colors = []
    for _ in range(n):
        color = tuple([np.random.randint(0, 256) for _ in range(3)])
        colors.append(color)
    return colors


def draw_bounding_boxes(
    image: np.ndarray,
    bboxes: np.ndarray,
    labels: Optional[List[str]] = None,
    colors: Optional[
        Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]
    ] = None,
    fill: Optional[bool] = False,
    width: int = 1,
    font: Optional[str] = None,
    font_size: Optional[int] = None,
) -> np.ndarray:

    """
    Draws bounding boxes on given image.
    The values of the input image should be uint8 between 0 and 255.
    If fill is True, Resulting Tensor should be saved as PNG image.

    Args:
        image (np.ndarray): Image with shape (H x W x C) to draw bounding boxes on.
        bboxes (np.ndarray): Bounding boxes with in unnormalized xywh format with shape (N x 4) or with shape (N x 6) if confidence scores and class labels are provided. Note that the boxes are absolute coordinates with respect to the image. In other words: `0 <= x|w <= W` and `0 <= y|h <= H`.
        labels (Optional[list[str]], optional): Labels for bounding boxes. Defaults to None.
        colors (Optional[Union[List[Union[str, Tuple[int, int, int]]], str, Tuple[int, int, int]]], optional): List containing the colors of the boxes or single color for all boxes. The color can be represented as PIL strings e.g. "red" or "#FF00FF", or as RGB tuples e.g. ``(240, 10, 157)``. By default, random colors are generated for boxes.
        fill (Optional[bool], optional): If true, fills bounding boxes with color. Defaults to False.
        width (int, optional): Width of bounding box lines. Defaults to 1.
        font (Optional[str], optional): Font to use for labels. Defaults to None.
        font_size (Optional[int], optional): Font size to use for labels. Defaults to None.

    Returns:
        img (np.ndarray[H, W, C]): Image ndarray of dtype uint8 with bounding boxes plotted.
    """

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Image must be of type np.ndarray. Got {type(image)}")
    if not isinstance(bboxes, np.ndarray):
        raise TypeError(
            f"Bounding boxes must be of type np.ndarray. Got {type(bboxes)}"
        )

    num_boxes = bboxes.shape[0]

    if num_boxes == 0:
        logger.warning("boxes doesn't contain any box. No box was drawn")
        return image

    if labels is None:
        labels: Union[List[str], List[None]] = [None] * num_boxes  # type: ignore[no-redef]
    elif len(labels) != num_boxes:
        raise ValueError(
            f"Number of boxes ({num_boxes}) and labels ({len(labels)}) mismatch. Please specify labels for each box."
        )

    if colors is None:
        colors = _generate_color_palette(num_boxes)
    elif isinstance(colors, list):
        if len(colors) < num_boxes:
            raise ValueError(
                f"Number of colors ({len(colors)}) is less than number of boxes ({num_boxes}). "
            )
    else:  # colors specifies a single color for all boxes
        colors = [colors] * num_boxes

    colors = [
        (ImageColor.getrgb(color) if isinstance(color, str) else color)
        for color in colors
    ]

    if font is None:
        if font_size is not None:
            logger.warning(
                "Argument 'font_size' will be ignored since 'font' is not set."
            )
        txt_font = ImageFont.load_default()
    else:
        try:
            txt_font = ImageFont.truetype(font=font, size=font_size or 10)
        except OSError:
            system_fonts = matplotlib.font_manager.findSystemFonts(
                fontpaths=None, fontext="ttf"
            )
            for i in range(len(system_fonts)):
                system_fonts[i] = system_fonts[i].split("/")[-1]
            logger.error(
                f"Font '{font}' not found. Select from the following fonts: \n{system_fonts}"
            )
            raise OSError

    # Handle Grayscale images
    if image.shape[2] == 1:
        image = image.repeat(3, 1, 1)

    ndarr = image
    img_to_draw = Image.fromarray(ndarr)
    img_boxes = bboxes.astype(np.int32)

    if fill:
        draw = ImageDraw.Draw(img_to_draw, "RGBA")
    else:
        draw = ImageDraw.Draw(img_to_draw)

    for bbox, color, label in zip(img_boxes, colors, labels):  # type: ignore[arg-type]
        x, y, w, h = bbox[:4]
        xyxy = (x, y, x + w, y + h)
        if fill:
            fill_color = color + (100,)
            draw.rectangle(xyxy, width=width, outline=color, fill=fill_color)
        else:
            draw.rectangle(xyxy, width=width, outline=color)

        if label is not None:
            margin = width + 1
            draw.text(
                (xyxy[0] + margin, xyxy[1] + margin), label, fill=color, font=txt_font
            )

    return np.array(img_to_draw)


def draw_tracks(
    img,
    tracks,
    fill: Optional[bool] = False,
    width: int = 1,
    font: Optional[str] = None,
    font_size: Optional[int] = None,
):
    colors = []
    for track in tracks:
        color = tuple([ord(c) * ord(c) % 256 for c in track.id[:3]])
        colors.append(color)

    bboxes = np.array([t.box for t in tracks])
    labels = [t.id[:3] for t in tracks]
    return draw_bounding_boxes(
        img, bboxes, labels, colors, fill, width, font, font_size
    )
