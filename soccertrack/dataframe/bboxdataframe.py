from __future__ import annotations

import random
from hashlib import md5
from typing import Any, Iterator, Optional, Type
from warnings import simplefilter

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..logging import logger
from ..utils import MovieIterator
from .base import SoccerTrackMixin

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# https://clrs.cc/
_COLOR_NAME_TO_RGB = dict(
    navy=((0, 38, 63), (119, 193, 250)),
    blue=((0, 120, 210), (173, 220, 252)),
    aqua=((115, 221, 252), (0, 76, 100)),
    teal=((15, 205, 202), (0, 0, 0)),
    olive=((52, 153, 114), (25, 58, 45)),
    green=((0, 204, 84), (15, 64, 31)),
    lime=((1, 255, 127), (0, 102, 53)),
    yellow=((255, 216, 70), (103, 87, 28)),
    orange=((255, 125, 57), (104, 48, 19)),
    red=((255, 47, 65), (131, 0, 17)),
    maroon=((135, 13, 75), (239, 117, 173)),
    fuchsia=((246, 0, 184), (103, 0, 78)),
    purple=((179, 17, 193), (241, 167, 244)),
    black=((24, 24, 24), (220, 220, 220)),
    gray=((168, 168, 168), (0, 0, 0)),
    silver=((220, 220, 220), (0, 0, 0)),
)
FRAME_INDEX = 0
BB_LEFT_INDEX = 1
BB_TOP_INDEX = 2
BB_RIGHT_INDEX = 3
BB_BOTTOM_INDEX = 4
BB_WIDTH_INDEX = 3
BB_HEIGHT_INDEX = 4
LABEL_INDEX = 7

_COLOR_NAMES = list(_COLOR_NAME_TO_RGB)


color_list = []
i = 0
while i < 1000:
    color_list.append(_COLOR_NAMES[random.randint(0, len(_COLOR_NAMES) - 1)])
    i += 1


def _rgb_to_bgr(color: tuple[int, ...]) -> list[Any]:
    """Convert RGB color to BGR color.

    Args:
        color (tuple): RGB color.

    Returns:
        list: BGR color.

    """
    return list(reversed(color))


class BBoxDataFrame(SoccerTrackMixin, pd.DataFrame):

    """Bounding box data frame.

    Args:
        pd.DataFrame (pd.DataFrame): Pandas DataFrame object.

    Returns:
        BBoxDataFrame: Bounding box data frame.

    Note:
        The bounding box data frame is a pandas DataFrame object with the following MultiIndex structure:
        Level 0: Team ID(str)
        Level 1: Player ID(str)
        Level 2: Attribute

        and the following attributes:
            frame (float): Frame ID.
            bb_left (float): Bounding box left coordinate.
            bb_top (float): Bounding box top coordinate.
            bb_width (float): Bounding box width.
            bb_height (float): Bounding box height.
            conf (float): Confidence of the bounding box.

        Since SoccerTrack basically only handles ball and person classes, class_id, etc.
        are not included in the BBoxDataframe for simplicity.
        However, they are needed for visualization and calculation of evaluation indicators,
        so they are generated as needed in additional attributes.
    """

    @property
    def _constructor(self: pd.DataFrame) -> Type[BBoxDataFrame]:
        """Return the constructor for the DataFrame.

        Args:
            self (pd.DataFrame): DataFrame object.

        Returns:
            BBoxDataFrame: BBoxDataFrame object.
        """
        return BBoxDataFrame

        # df: pd.DataFrame

    def get_id(self: BBoxDataFrame) -> list[tuple[Any, Any]]:
        """Get the team id and player id of the DataFrame.

        Args:
            self (BBoxDataFrame): BBoxDataFrame object.

        Returns:
            id_list(list[tuple[Any, Any]]): List of team id and player id.
        """
        id_list = []
        for column in self.columns:
            team_id = column[0]
            player_id = column[1]
            id_list.append((team_id, player_id))
        return id_list
    
    def visualize_frame(
        self: BBoxDataFrame, frame_idx: int, frame: np.ndarray
    ) -> np.ndarray:
        """Visualize the bounding box of the specified frame.

        Args:
            self (BBoxDataFrame): BBoxDataFrame object.
            frame_idx (int): Frame ID.
            frame (np.ndarray): Frame image.
        Returns:
            frame(np.ndarray): Frame image with bounding box.
        """
        self.index: pd.MultiIndex

        frame_df = self.loc[self.index == frame_idx].copy()
        for index, group in frame_df.groupby(level=("TeamID", "PlayerID"), axis=1):
            group_stack = group.to_long_df().reset_index()
            #Check if pandas lines do not contain nan
            if not group_stack.isnull().values.any():
                TeamID, PlayerID = index
                x1, y1, w, h = group_stack[['bb_left', 'bb_top', 'bb_width', 'bb_height']].values[0]
                logger.debug(
                f"x:{int(x1)}, y:{int(y1)}, x2:{int(x1 + w)}, y2:{int(y1 + h)}, label:{TeamID}-{PlayerID}, color:{color_list[int(PlayerID)]}"
                )
                frame = add_bbox_to_frame(
                    frame, int(x1), int(y1), int(x1 + w), int(y1 + h), label=f"{TeamID}-{PlayerID}", color=color_list[int(PlayerID)]
                )
        return frame


    def visualize_frames(self, video_path: str) -> Iterator[np.ndarray]:
        """Visualize bounding boxes on a video.

        Args:
            video_path (str): Path to the video file.

        Returns:
            None
        """

        movie_iterator = MovieIterator(video_path)
        for frame_idx, frame in tqdm(enumerate(movie_iterator)):
            img_ = self.visualize_frame(frame_idx, frame)
            yield img_

def add_bbox_to_frame(
    image: np.ndarray,
    left: int,
    top: int,
    right: int,
    bottom: int,
    label: Optional[str] = None,
    color: Optional[str] = None,
) -> np.ndarray:
    """Add bounding box and label to image.

    Args:
        img (np.ndarray): Image.
        left (int): Bounding box left coordinate.
        top (int): Bounding box top coordinate.
        right (int): Bounding box right coordinate.
        bottom (int): Bounding box bottom coordinate.
        label (str): Label.
        color (str): Color.
    Returns:
        img (np.ndarray): Image with bounding box and label.
    """
    _DEFAULT_COLOR_NAME = "purple"

    if isinstance(image, np.ndarray) is False:
        raise TypeError("'image' parameter must be a numpy.ndarray")
    try:
        left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    except ValueError as e:
        raise TypeError("'left', 'top', 'right' & 'bottom' must be a number") from e

    if label and isinstance(label, str) is False:
        raise TypeError("'label' must be a str")

    if label and not color:
        hex_digest = md5(label.encode()).hexdigest()
        color_index = int(hex_digest, 16) % len(_COLOR_NAME_TO_RGB)
        color = _COLOR_NAMES[color_index]

    if not color:
        color = _DEFAULT_COLOR_NAME

    if isinstance(color, str) is False:
        raise TypeError("'color' must be a str")

    if color not in _COLOR_NAME_TO_RGB:
        msg = "'color' must be one of " + ", ".join(_COLOR_NAME_TO_RGB)
        raise ValueError(msg)

    colors = [_rgb_to_bgr(item) for item in _COLOR_NAME_TO_RGB[color]]
    color_value, _ = colors

    image = cv2.rectangle(image, (left, top), (right, bottom), color_value, 2)

    if label:

        _, image_width, _ = image.shape
        fontface = cv2.FONT_HERSHEY_TRIPLEX
        fontscale = 0.5
        thickness = 1

        (label_width, label_height), _ = cv2.getTextSize(
            label, fontface, fontscale, thickness
        )
        rectangle_height, rectangle_width = 1 + label_height, 1 + label_width

        rectangle_bottom = top
        rectangle_left = max(0, min(left - 1, image_width - rectangle_width))

        rectangle_top = rectangle_bottom - rectangle_height
        rectangle_right = rectangle_left + rectangle_width

        label_top = rectangle_top + 1

        if rectangle_top < 0:
            rectangle_top = top
            rectangle_bottom = rectangle_top + label_height + 1

            label_top = rectangle_top

        label_left = rectangle_left + 1
        label_bottom = label_top + label_height
        # _ = label_left + label_width

        rec_left_top = (rectangle_left, rectangle_top)
        rec_right_bottom = (rectangle_right, rectangle_bottom)

        cv2.rectangle(image, rec_left_top, rec_right_bottom, color_value, -1)

        # image[label_top:label_bottom, label_left:label_right, :] = label

        cv2.putText(
            image,
            text=label,
            org=(label_left, int((label_bottom))),
            fontFace=fontface,
            fontScale=fontscale,
            color=(0, 0, 0),
            thickness=thickness,
            lineType=cv2.LINE_4,
        )
    return image
