from __future__ import annotations

import uuid
from hashlib import md5
from pathlib import Path
from typing import Any, Iterable, Optional, Type

import cv2
import numpy as np
import pandas as pd

from soccertrack.utils import make_video

from ..logger import logger
from ..utils import MovieIterator, get_fps
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

# global index to access bbox information for each frame
X_INDEX = 0  # xmin
Y_INDEX = 1  # ymin
W_INDEX = 2  # width
H_INDEX = 3  # height
CONFIDENCE_INDEX = 4
CLASS_ID_INDEX = 5
IMAGE_NAME_INDEX = 6
OBJECT_ID_INDEX = 7


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

    def visualize_frame(
        self: BBoxDataFrame,
        frame_idx: int,
        frame: np.ndarray,
        draw_frame_id: bool = False,
    ) -> np.ndarray:
        """Visualize the bounding box of the specified frame.

        Args:
            self (BBoxDataFrame): BBoxDataFrame object.
            frame_idx (int): Frame ID.
            frame (np.ndarray): Frame image.
            draw_frame_id (bool, optional): Whether to draw the frame ID. Defaults to False.
        Returns:
            frame(np.ndarray): Frame image with bounding box.
        """
        if frame_idx not in self.index:
            return frame
        frame_df = self.loc[self.index == frame_idx]

        for (team_id, player_id), player_df in frame_df.iter_players():
            if player_df.isnull().any(axis=None):
                logger.debug(
                    f"NaN value found at frame {frame_idx}, team {team_id}, player {player_id}. Skipping..."
                )
                continue

            logger.debug(
                f"Visualizing frame {frame_idx}, team {team_id}, player {player_id}"
            )
            if frame_idx not in player_df.index:
                logger.debug(f"Frame {frame_idx} not found in player_df")
                continue

            attributes = player_df.loc[
                frame_idx, ["bb_left", "bb_top", "bb_width", "bb_height"]
            ]

            x1, y1, w, h = player_df.loc[
                frame_idx, ["bb_left", "bb_top", "bb_width", "bb_height"]
            ].values.astype(int)
            x2, y2 = x1 + w, y1 + h

            label = f"{team_id}_{player_id}"
            player_id_int = sum([int(x) for x in str(hash(player_id))[1:]])
            color = _COLOR_NAMES[hash(player_id_int) % len(_COLOR_NAMES)]

            logger.debug(
                f"x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}, label: {label}, color: {color}"
            )
            frame = add_bbox_to_frame(frame, x1, y1, x2, y2, label, color)

        if draw_frame_id:
            frame = add_frame_id_to_frame(frame, frame_idx)
        return frame

    def visualize_frames(self, video_path: str, save_path: str, **kwargs) -> None:
        """Visualize bounding boxes on a video.

        Args:
            video_path (str): Path to the video file.

        Returns:
            None
        """

        def generator():
            movie_iterator = MovieIterator(video_path)
            for frame_idx, frame in zip(self.index, movie_iterator):
                img_ = self.visualize_frame(frame_idx, frame)
                yield img_

        input_framerate = get_fps(video_path)

        make_video(generator(), save_path, input_framerate=input_framerate, **kwargs)

    def to_yolo_format(self):
        """Convert a dataframe to the YOLO format.

        Returns:
            pd.DataFrame: Dataframe in YOLO format.
        """
        raise NotImplementedError

    def to_yolov5_format(
        self,
        mapping: dict[dict[Any, Any], dict[Any, Any]] = None,
        na_class: int = 0,
        h: int = None,
        w: int = None,
        save_dir: str = None,
    ):
        """Convert a dataframe to the YOLOv5 format.

        Converts a dataframe to the YOLOv5 format. The specification for each line is as follows:
        <class_id> <x_center> <y_center> <width> <height>

        * One row per object
        * Each row is class x_center y_center width height format.
        * Box coordinates must be normalized by the dimensions of the image (i.e. have values between 0 and 1)
        * Class numbers are zero-indexed (start from 0).

        Args:
            mapping (dict, optional): Mappings from team_id and player_id to class_id. Should contain one or two nested dictionaries like {'TeamID':{0:1}, 'PlayerID':{0:1}}. Defaults to None. If None,the class_id will be inferred from the team_id and player_id and set such that players=0 and ball=1.
            na_class (int, optional): Class ID for NaN values. Defaults to 0.
            h (int, optional): Height of the image. Unnecessary if the dataframe has height metadata. Defaults to None.
            w (int, optional): Width of the image. Unnecessary if the dataframe has width metadata. Defaults to None.
            save_dir (str, optional): If specified, saves a text file for each frame in the specified directory. Defaults to None.
        Returns:
            list: list of shape (N, M, 5) in YOLOv5 format. Where N is the number of frames, M is the number of objects in the frame, and 5 is the number of attributes (class_id, x_center, y_center, width, height).
        """

        if save_dir is not None:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)

        df = self.to_long_df().reset_index()

        if mapping is None:
            mapping = {"TeamID": {"3": 1}, "PlayerID": {}}

        team_mappings = df["TeamID"].map(mapping["TeamID"])
        player_mappings = df["PlayerID"].map(mapping["PlayerID"])
        df["class"] = (
            player_mappings.combine_first(team_mappings).fillna(na_class).astype(int)
        )

        df["x"] = df["bb_left"] + df["bb_width"] / 2
        df["y"] = df["bb_top"] + df["bb_height"] / 2

        return_values = []
        groups = df.groupby("frame")
        for frame_num, group in groups:
            vals = group[["class", "x", "y", "bb_width", "bb_height"]].values
            vals /= np.array([1, w, h, w, h])

            return_values.append(vals)

            if save_dir is not None:
                filename = f"{frame_num:06d}.txt"
                save_path = save_dir / filename
                np.savetxt(save_path, vals, fmt="%d %f %f %f %f")

        return return_values

    def to_mot_format(self):
        """Convert a dataframe to the MOT format.

        Returns:
            pd.DataFrame: Dataframe in MOT format.
        """
        raise NotImplementedError

    def to_labelbox_segment(self: BBoxDataFrame) -> dict:
        """Convert a dataframe to the Labelbox segment format.

        Args:
            self (BBoxDataFrame): BBoxDataFrame object.

        Returns:
            segment: Dictionary in Labelbox segment format.

        Notes:
            The Labelbox segment format is a dictionary with the following structure:
            {feature_name:
                {keyframes:
                    {frame:
                        {bbox:
                            {top: XX,
                            left: XX,
                            height: XX,
                            width: XX},
                        label: label
                        }
                    },
                    {frame:
                    ...

                    }
                }
            }
        """
        segment = dict()
        for (team_id, player_id), player_bbdf in self.iter_players():
            feature_name = f"{team_id}_{player_id}"
            key_frames_dict = dict()
            key_frames_dict["keyframes"] = []
            missing_bbox = 0

            for idx, row in player_bbdf.iterrows():
                # Processing when player_bbdf contains no data
                try:
                    key_frames_dict["keyframes"].append(
                        {
                            "frame": idx + 1,
                            "bbox": {
                                "top": int(row["bb_top"]),
                                "left": int(row["bb_left"]),
                                "height": int(row["bb_height"]),
                                "width": int(row["bb_width"]),
                            },
                        }
                    )
                except ValueError as e:
                    missing_bbox += 1

            if missing_bbox > 0:
                logger.warning(
                    f"Missing {missing_bbox} bounding boxes for {feature_name}"
                )
            segment[feature_name] = [key_frames_dict]
        return segment

    def to_labelbox_data(
        self: BBoxDataFrame, 
        data_row : object, 
        schema_lookup : dict
        ) -> list:
        """Convert a dataframe to the Labelbox format.

        Args:
            self (BBoxDataFrame): BBoxDataFrame object.
            data_row (DataRow): DataRow object.
            schema_lookup(dict): Dictionary of label names and label ids.
        
        Returns:
            uploads(list): List of dictionaries in Labelbox format.

        """
        # convert to labelbox segment format
        segment = self.to_labelbox_segment()

        uploads = []
        for schema_name, schema_id in schema_lookup.items():
            if schema_name in segment:
                uploads.append(
                    {
                        "uuid": str(uuid.uuid4()),
                        "schemaId": schema_id,
                        "dataRow": {"id": data_row.uid},
                        "segments": segment[schema_name],
                    }
                )

        return uploads

    def to_list_of_tuples_format(
        self,
        mapping: dict[dict[Any, Any], dict[Any, Any]] = None,
        na_class: int | str = "player",
    ):
        """Convert a dataframe to a list of tuples.

        Converts a dataframe to a list of tuples necessary for calculating object detection metrics such as mAP and AP scores. The specification for each list element is as follows:
        (x, y, w, h, confidence, class_id, image_name, object_id)

        Returns:
            list: List of tuples.
        """
        long_df = self.to_long_df().reset_index()

        # TODO: Abstract mapping functionality to a separate function

        if mapping is None:
            mapping = {"TeamID": {"3": "ball"}, "PlayerID": {}}

        team_mappings = long_df["TeamID"].map(mapping["TeamID"])
        player_mappings = long_df["PlayerID"].map(mapping["PlayerID"])
        long_df["class"] = player_mappings.combine_first(team_mappings).fillna(na_class)
        long_df["image_name"] = long_df["frame"].astype(
            int
        )  # TODO: This won't work for object detection
        long_df["object_id"] = (
            long_df["PlayerID"].astype(str) + "_" + long_df["TeamID"].astype(str)
        )

        # create a unique object id for each object in ascending order
        assigned_ids = {
            p_id_t_id: object_id
            for object_id, p_id_t_id in enumerate(long_df["object_id"].unique())
        }
        long_df["object_id"] = long_df["object_id"].map(assigned_ids).astype(int)

        cols = [
            "bb_left",
            "bb_top",
            "bb_width",
            "bb_height",
            "conf",
            "class",
            "image_name",
            "object_id",
        ]
        return long_df[cols].values

    def preprocess_for_mot_eval(self):
        """Preprocess a dataframe for evaluation using the MOT metrics.

        Args:
            self (BBoxDataFrame): BBoxDataFrame object.

        Returns:
            ids (list): List of lists of object ids for each frame.
            dets (list): A list of arrays of detections in the format (x, y, w, h) for each frame.
        """

        if self.size == 0:
            return [], []

        # make a list of lists such that each list contains the detections for a single frame
        list_of_tuples = self.to_list_of_tuples_format()

        list_of_list_of_bboxes = np.split(
            list_of_tuples,
            np.unique(list_of_tuples[:, IMAGE_NAME_INDEX], return_index=True)[1][1:],
        )

        frame_idxs = []

        for list_of_bboxes in list_of_list_of_bboxes:
            try:
                frame_idxs.append(
                    list_of_bboxes[:, IMAGE_NAME_INDEX].astype("int64")[0]
                )
            except IndexError:
                frame_idxs.append(None)

        ids = [
            list_of_bboxes[:, OBJECT_ID_INDEX].astype("int64")
            for list_of_bboxes in list_of_list_of_bboxes
        ]

        dets = [
            list_of_bboxes[:, [X_INDEX, Y_INDEX, W_INDEX, H_INDEX]].astype("int64")
            for list_of_bboxes in list_of_list_of_bboxes
        ]

        start_frame = self.index.min()
        end_frame = self.index.max()
        missing_frames = np.setdiff1d(range(start_frame, end_frame), frame_idxs)

        # add empty detections for missing frames
        for missing_frame in missing_frames:
            # index to insert is not always the same as the missing frame index
            # example. if starting frame is 10 and missing frame is 12, insert index is 2
            insert_index = missing_frame - start_frame
            ids.insert(insert_index, np.array([]))
            dets.insert(insert_index, np.array([]))

        return ids, dets

    @staticmethod
    def from_dict(
        d: dict,
        attributes: Optional[Iterable[str]] = (
            "bb_left",
            "bb_top",
            "bb_width",
            "bb_height",
        ),
    ):
        """Create a BBoxDataFrame from a nested dictionary contating the coordinates of the players and the ball.

        The input dictionary should be of the form:
        {
            home_team_key: {
                PlayerID: {frame: [x, y], ...},
                PlayerID: {frame: [x, y], ...},
                ...
            },
            away_team_key: {
                PlayerID: {frame: [x, y], ...},
                PlayerID: {frame: [x, y], ...},
                ...
            },
            ball_key: {
                frame: [x, y],
                frame: [x, y],
                ...
            }
        }
        The `PlayerID` can be any unique identifier for the player, e.g. their jersey number or name. The PlayerID for the ball can be omitted, as it will be set to "0". `frame` must be an integer identifier for the frame number.

        Args:
            dict (dict): Nested dictionary containing the coordinates of the players and the ball.
            attributes (Optional[Iterable[str]], optional): Attributes to use for the coordinates. Defaults to ("x", "y").

        Returns:
            CoordinatesDataFrame: CoordinatesDataFrame.
        """
        attributes = list(attributes)  # make sure attributes is a list

        data = []
        for team, team_dict in d.items():
            for player, player_dict in team_dict.items():
                for frame, bbox in player_dict.items():
                    data.append([team, player, frame, *bbox])

        df = pd.DataFrame(
            data,
            columns=["TeamID", "PlayerID", "frame", *attributes],
        )

        df.pivot(index="frame", columns=["TeamID", "PlayerID"], values=attributes)
        df = df.pivot(index="frame", columns=["TeamID", "PlayerID"], values=attributes)
        multi_index = pd.MultiIndex.from_tuples(
            df.columns.swaplevel(0, 1).swaplevel(1, 2)
        )
        df.columns = pd.MultiIndex.from_tuples(multi_index)
        df.rename_axis(["TeamID", "PlayerID", "Attributes"], axis=1, inplace=True)
        df.sort_index(axis=1, inplace=True)

        return BBoxDataFrame(df)


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

        rec_left_top = (rectangle_left, rectangle_top)
        rec_right_bottom = (rectangle_right, rectangle_bottom)

        cv2.rectangle(image, rec_left_top, rec_right_bottom, color_value, -1)

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


def add_frame_id_to_frame(image: np.ndarray, frame_id: int) -> np.ndarray:
    """Add frame id to image.

    Args:
        img (np.ndarray): Image.
        frame_id (int): Frame id.
    Returns:
        img (np.ndarray): Image with frame id.
    """
    if isinstance(image, np.ndarray) is False:
        raise TypeError("'image' parameter must be a numpy.ndarray")

    try:
        frame_id = int(frame_id)
    except ValueError as e:
        raise TypeError("'frame_id' must be a number") from e

    fontface = cv2.FONT_HERSHEY_TRIPLEX
    fontscale = 5
    thickness = 1

    label = f"frame: {frame_id}"

    # draw frame id on the top left corner
    (label_width, label_height), _ = cv2.getTextSize(
        label, fontface, fontscale, thickness
    )

    label_left = 10
    label_bottom = label_height + 10

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
