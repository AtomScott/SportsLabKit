from __future__ import annotations

import random
from hashlib import md5
from typing import Any, Iterator, Optional, Type

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from ..utils import MovieIterator
from .base import SoccerTrackMixin

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

_COLOR_NAMES = list(_COLOR_NAME_TO_RGB)


def _rgb_to_bgr(color: tuple[int, ...]) -> list[Any]:
    """Convert RGB color to BGR color.

    Args:
        color (tuple): RGB color.

    Returns:
        list: BGR color.

    """
    return list(reversed(color))


BB_LEFT_INDEX = 1
BB_TOP_INDEX = 2
BB_RIGHT_INDEX = 3
BB_BOTTOM_INDEX = 4
BB_WIDTH_INDEX = 3
BB_HEIGHT_INDEX = 4

color_list = []
i = 0
while i < 1000:
    color_list.append(_COLOR_NAMES[random.randint(0, len(_COLOR_NAMES) - 1)])
    i += 1


class BBoxDataFrame(SoccerTrackMixin, pd.DataFrame):
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

    def to_list(self: BBoxDataFrame, xywh: bool = True) -> list[list[float]]:
        """Convert a dataframe column to a 2-dim list for evaluation of object detection.

        Args:
            df (BBoxDataFrame): BBoxDataFrame
            xywh (bool): If True, convert to x1y1x2y2 format. Defaults to True.
        Returns:
            bbox_2dim_list(list): 2-dim list
        Note:
            Description of each element in bbox_2dim_list.
            The elements of the final included bbox_2dim_list are as follows:
            bbox_cols(list) : ['frame_id', 'bb_left', 'bb_top', 'bb_right', 'bb_bottom','conf', 'class_id' ,'obj_id']
                frame_id(int) : Frame ID
                bb_left(float) : Bounding box left coordinate
                bb_top(float) : Bounding box top coordinate
                bb_right(float) : Bounding box right coordinate
                bb_bottom(float) : Bounding box bottom coordinate
                conf(float) : Confidence score
                class_id(int) : Class ID. In the normal case, follow the value of the coco image dataset.
                obj_id(int) : Object ID. This id is a unique integer value for each track
                            (corresponding to each column in the BBoxDataframe) that is used when evaluating tracking.
            The BBoxDataframe input to this function contains information other than frame_id and obj_id in advance.
            This function adds the obj_id column and converts the BBoxDataframe to a list.
        """

        # append_object_id
        obj_id = 0
        id_list = []
        for column in self.columns:
            team_id = column[0]
            player_id = column[1]
            id_list.append((team_id, player_id))
        for id in sorted(set(id_list)):
            self.loc[
                :, (id[0], id[1], "obj_id")
            ] = obj_id  # Add obj_id column for each player's Attribute
            obj_id += 1
        df_sort = self[
            self.sort_index(axis=1, level=[0, 1], ascending=[True, True]).columns
        ]
        # Create two-dimensional lists from dataframes
        bbox_2dim_list = []
        bbox_cols = [
            "bb_left",
            "bb_top",
            "bb_width",
            "bb_height",
            "conf",
            "class_id",
            "obj_id",
        ]
        num_cols = len(bbox_cols)
        df2list = df_sort.values.tolist()

        for frame_id, frame_raw in enumerate(df2list):
            for idx in range(0, len(frame_raw), num_cols):
                if not any(
                    pd.isnull(frame_raw[idx : idx + num_cols])
                ):  # extract nan rows
                    bbox_2dim_list.append([frame_id] + frame_raw[idx : idx + num_cols])

        # bbox_2dim_list = [x for x in bbox_2dim_list if not any(pd.isnull(x))]

        for bbox in bbox_2dim_list:
            if xywh:
                bbox[BB_RIGHT_INDEX] = bbox[BB_WIDTH_INDEX] + bbox[BB_LEFT_INDEX]
                bbox[BB_BOTTOM_INDEX] = bbox[BB_HEIGHT_INDEX] + bbox[BB_TOP_INDEX]

        return bbox_2dim_list

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

        BB_LEFT_INDEX = 0
        BB_TOP_INDEX = 1
        BB_WIDTH_INDEX = 2
        BB_HEIGHT_INDEX = 3

        unique_team_ids = self.columns.get_level_values("TeamID").unique()
        for team_id in unique_team_ids:
            bboxdf_team = self.xs(team_id, level="TeamID", axis=1)

            unique_team_ids = bboxdf_team.columns.get_level_values("PlayerID").unique()
            for idx, player_id in enumerate(unique_team_ids):
                color = color_list[idx]

                bboxdf_player = bboxdf_team.xs(player_id, level="PlayerID", axis=1)
                bboxdf_player = bboxdf_player.reset_index(drop=True)
                bbox = bboxdf_player.loc[frame_idx]
                if (
                    np.isnan(bbox[BB_LEFT_INDEX]) is False
                    and np.isnan(bbox[BB_TOP_INDEX]) is False
                ):
                    bb_left = bbox[BB_LEFT_INDEX]
                    bb_top = bbox[BB_TOP_INDEX]
                    bb_right = bb_left + bbox[BB_WIDTH_INDEX]
                    bb_bottom = bb_top + bbox[BB_HEIGHT_INDEX]

                    x, y, x2, y2 = list(
                        [int(bb_left), int(bb_top), int(bb_right), int(bb_bottom)]
                    )
                    frame = add(
                        frame, x, y, x2, y2, label=f"{team_id}_{player_id}", color=color
                    )

        return frame

    def visualize_bbox(self, video_path: str) -> Iterator[np.ndarray]:
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


def add(
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

    image = cv2.rectangle(image, (left, top), (right, bottom), color, 2)

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
