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

    def to_list(self: BBoxDataFrame, xywh: bool = True) -> Any:
        """Convert a dataframe column to a 2-dim list for evaluation of object detection.

        Args:
            self (BBoxDataFrame): BBoxDataFrame
            xywh (bool): If True, convert to x1y1x2y2 format. Defaults to True.
        Returns:
            bbox_2dim_list(Any): 2-dim list of bounding boxes.
        Note:
            Description of each element in bbox_2dim_list.
            The elements of the final included bbox_2dim_list are as follows:
            bbox_cols(list) : ['frame', 'bb_left', 'bb_top', 'bb_right', 'bb_bottom','conf', 'class_id' ,'obj_id']
                frame(int) : Frame ID
                bb_left(float) : Bounding box left coordinate
                bb_top(float) : Bounding box top coordinate
                bb_right(float) : Bounding box right coordinate
                bb_bottom(float) : Bounding box bottom coordinate
                conf(float) : Confidence score
                class_id(int) : Class ID. In the normal case, follow the value of the coco image dataset.
                "team_player_id(str): Team ID and player ID. The format is "{team_id}_{player_id}".
                                        This is primarily used for labels in the bbox visualization.
                obj_id(int) : Object ID. This id is a unique integer value for each track
                            (corresponding to each column in the BBoxDataframe) that is used when evaluating tracking.
                            team_player_id is str, and the built-in evaluation indicator does not support the type of str,
                            so a unique id of type float is created separately.

            The BBoxDataframe input to this function contains information other than frame, class_id, team_player_id, and obj_id in advance.
            This function adds some columns and converts the BBoxDataframe to a list.

            In order to avoid the pandas PerformanceWarning that occurs in this method, a simplefilter method is introduced (see. line 17.) 
            This is a warning derived from the pandas version and does not directly affect the output, so the warning is implicitly stopped.
        """

        bbox_cols = [
            "frame",
            "bb_left",
            "bb_top",
            "bb_width",
            "bb_height",
            "conf",
            "class_id",
            "team_player_id",
            "obj_id",
        ]
        # get team_id and player_id
        id_list = self.get_id()
        # append_object_id
        obj_id = 0

        for id in sorted(set(id_list)):
            if id[0] == "3":
                self.loc[:, (id[0], id[1], "class_id")] = 32  # sports ball
            else:
                self.loc[:, (id[0], id[1], "class_id")] = 0  # person

            team_player_id = f"{id[0]}-{id[1]}"

            self.loc[
                :, (id[0], id[1], "team_player_id")
            ] = team_player_id  # Add team_player_id column for each player's Attribute

            self.loc[
                :, (id[0], id[1], "obj_id")
            ] = obj_id  # Add obj_id column for each player's Attribute

            obj_id += 1
        df_sort = self[
            self.sort_index(axis=1, level=[0, 1], ascending=[True, True]).columns
        ]

        df_stack = (
            df_sort.stack(("TeamID", "PlayerID"))
            .reset_index()
            .drop(columns=["TeamID", "PlayerID"], axis=1)
        )
        bbox_2dim = df_stack.reindex(columns=bbox_cols).values
        bbox_2dim_list = bbox_2dim[~pd.isna(bbox_2dim).any(axis=1)].tolist()
        frame_idx_start = bbox_2dim_list[0][FRAME_INDEX]
        for bbox in bbox_2dim_list:
            bbox[FRAME_INDEX] = bbox[FRAME_INDEX] - frame_idx_start  # start from 0
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
        self.index: pd.MultiIndex

        if self.index[0] != 0:
            self.index = self.index - self.index[0]
        frame_df = self.loc[self.index == frame_idx].copy()
        bboxes = frame_df.to_list()
        for col_ids, bbox in enumerate(bboxes):
            color = color_list[col_ids]
            x, y, x2, y2 = bbox[BB_LEFT_INDEX : BB_BOTTOM_INDEX + 1]
            label = bbox[LABEL_INDEX]
            logger.debug(
                f"x:{x}, y:{y}, x2:{x2}, y2:{y2}, label:{label}, color:{color}"
            )
            frame = add(
                frame, int(x), int(y), int(x2), int(y2), label=label, color=color
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
        frame_idx_list = []
        for frame_idx in range(len(self)):
            frame_idx_list.append(frame_idx)

        # img_list = []
        for frame_idx, frame in tqdm(zip(frame_idx_list[:-1], movie_iterator)):
            img_ = self.visualize_frame(frame_idx, frame)
            yield img_
        #     img_list.append(img_)
        # return img_list


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
