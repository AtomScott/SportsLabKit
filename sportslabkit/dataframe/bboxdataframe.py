from __future__ import annotations

import uuid
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from sportslabkit.utils import make_video
from sportslabkit.viz.visualizers import BaseVisualizer, SimpleVisualizer

from ..logger import logger
from ..utils import MovieIterator, get_fps
from .base import BaseSLKDataFrame
from .coordinatesdataframe import CoordinatesDataFrame


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


class BBoxDataFrame(BaseSLKDataFrame):
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
    def _constructor(self: pd.DataFrame) -> type[BBoxDataFrame]:
        """Return the constructor for the DataFrame.

        Args:
            self (pd.DataFrame): DataFrame object.

        Returns:
            BBoxDataFrame: BBoxDataFrame object.
        """
        return BBoxDataFrame

    def visualize_frame(
        self,
        frame_idx: int,
        frame: np.ndarray,
        visualizer:BaseVisualizer | None = None,
    ) -> np.ndarray:
        """Visualize the bounding box of the specified frame.

        Args:
            self (BBoxDataFrame): BBoxDataFrame object.
            frame_idx (int): Frame ID.
            frame (np.ndarray): Frame image.
            visualizer (BaseVisualizer, optional): Visualizer object. If None, SimpleVisualizer is used. Defaults to None.

        Returns:
            np.ndarray: Frame image with bounding box if frame_idx is in the index, otherwise returns the original frame image.
        """

        if visualizer is None:
            visualizer = SimpleVisualizer()

        if frame_idx not in self.index:
            return frame

        frame_df = self.loc[self.index == frame_idx]
        drawn_frame = visualizer.draw_frame(frame_df, frame)
        return drawn_frame

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
        mapping: dict[dict[Any, Any], dict[Any, Any]] | None = None,
        na_class: int = 0,
        h: int | None = None,
        w: int | None = None,
        save_dir: str | None = None,
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
        df["class"] = player_mappings.combine_first(team_mappings).fillna(na_class).astype(int)

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

        df = self.to_long_df()

        # Reset the index
        df.reset_index(inplace=True)

        # grab all unique TeamIDs
        team_ids = df.TeamID.unique()
        # grab all unique PlayerIDs for each TeamID
        player_ids = {team_id: df[df.TeamID == team_id].PlayerID.unique() for team_id in team_ids}

        # create a mapping from TeamID and PlayerID to a unique integer id
        id_map = {}
        num_ids = 0
        for team_id in team_ids:
            id_map[team_id] = {}
            for player_id in player_ids[team_id]:
                id_map[team_id][player_id] = num_ids
                num_ids += 1
        logger.debug(f"Using the following id_map: {id_map}")

        # Create the 'id' column
        df["id"] = df.apply(lambda row: id_map[row["TeamID"]][row["PlayerID"]], axis=1)

        # Select the desired columns
        df = df[["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height", "conf"]]

        # add the x, y, z columns (all -1)
        df = df.assign(x=-1, y=-1, z=-1)
        df = df.sort_values(by=["frame", "id"])

        # check that each frame contains only unique ids
        dupes = df[df.duplicated(subset=["frame", "id"])]
        if not dupes.empty:
            raise ValueError(f"Duplicate ids found in the following frames: {dupes.frame.unique()}")

        return df

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
        segment = {}
        for (team_id, player_id), player_bbdf in self.iter_players():
            feature_name = f"{team_id}_{player_id}"
            key_frames_dict = {}
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
                except ValueError:
                    missing_bbox += 1

            if missing_bbox > 0:
                logger.debug(f"Missing {missing_bbox} bounding boxes for {feature_name}")
            segment[feature_name] = [key_frames_dict]
        return segment

    def to_labelbox_data(self: BBoxDataFrame, data_row: object, schema_lookup: dict) -> list:
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
        mapping: dict[dict[Any, Any], dict[Any, Any]] | None = None,
        na_class: int | str = "player",
    ):
        """Convert a dataframe to a list of tuples.

        Converts a dataframe to a list of tuples necessary for calculating object detection metrics such as mAP and AP scores. The specification for each list element is as follows:
        (x, y, w, h, confidence, class_id, image_name, object_id)

        Returns:
            list: List of tuples.
        """
        long_df = self.to_long_df().reset_index()
        # drop nan rows
        long_df = long_df.dropna()

        # TODO: Abstract mapping functionality to a separate function

        if mapping is None:
            mapping = {"TeamID": {"3": "ball"}, "PlayerID": {}}

        team_mappings = long_df["TeamID"].map(mapping["TeamID"])
        player_mappings = long_df["PlayerID"].map(mapping["PlayerID"])
        long_df["class"] = player_mappings.combine_first(team_mappings).fillna(na_class)
        long_df["image_name"] = long_df["frame"].astype(int)  # TODO: This won't work for object detection
        long_df["object_id"] = long_df["PlayerID"].astype(str) + "_" + long_df["TeamID"].astype(str)

        # create a unique object id for each object in ascending order
        assigned_ids = {p_id_t_id: object_id for object_id, p_id_t_id in enumerate(long_df["object_id"].unique())}
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

    def to_codf(self: BBoxDataFrame, H: np.ndarray, method: str = "bottom_middle") -> CoordinatesDataFrame:
        """
        Converts bounding box dataframe to a new coordinate dataframe using a given homography matrix.

        This function takes a dataframe of bounding boxes and applies a perspective transformation
        to a specified point within each bounding box (e.g., center, bottom middle, top middle) into
        a new coordinate frame (e.g., a pitch coordinate frame). The result is returned as a
        CoordinatesDataFrame.

        Args:
            self (BBoxDataFrame): A dataframe containing bounding box coordinates.
            H (np.ndarray): A 3x3 homography matrix used for the perspective transformation.
            method (str): Method to determine the point within the bounding box to transform.
                        Options include 'center', 'bottom_middle', 'top_middle'.

        Returns:
            CoordinatesDataFrame: A dataframe containing the transformed coordinates.

        Example:
            H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
            bbox_data = BBoxDataFrame(...)
            codf = bbox_data.to_codf(H, method='bottom_middle')
        """
        assert H.shape == (3, 3), "H must be a 3x3 matrix"

        long_df = self.to_long_df()

        if method == "center":
            long_df["x"] = long_df["bb_left"] + long_df["bb_width"] / 2
            long_df["y"] = long_df["bb_top"] + long_df["bb_height"] / 2
        elif method == "bottom_middle":
            long_df["x"] = long_df["bb_left"] + long_df["bb_width"] / 2
            long_df["y"] = long_df["bb_top"] + long_df["bb_height"]
        elif method == "top_middle":
            long_df["x"] = long_df["bb_left"] + long_df["bb_width"] / 2
            long_df["y"] = long_df["bb_top"]
        else:
            raise ValueError("Invalid method. Options are 'center', 'bottom_middle', 'top_middle'.")

        pts = long_df[["x", "y"]].values
        pitch_pts = cv2.perspectiveTransform(np.asarray([pts], dtype=np.float32), H)
        long_df["x"] = pitch_pts[0, :, 0]
        long_df["y"] = pitch_pts[0, :, 1]

        codf = CoordinatesDataFrame(
            long_df[["x", "y"]]
            .unstack(level=["TeamID", "PlayerID"])
            .reorder_levels([1, 2, 0], axis=1)
            .sort_index(axis=1)
        )
        return codf

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
                frame_idxs.append(list_of_bboxes[:, IMAGE_NAME_INDEX].astype("int64")[0])
            except IndexError:
                frame_idxs.append(None)

        ids = [list_of_bboxes[:, OBJECT_ID_INDEX].astype("int64") for list_of_bboxes in list_of_list_of_bboxes]

        dets = [
            list_of_bboxes[:, [X_INDEX, Y_INDEX, W_INDEX, H_INDEX]].astype("int64")
            for list_of_bboxes in list_of_list_of_bboxes
        ]

        start_frame = self.index.min()
        end_frame = self.index.max()
        missing_frames = np.setdiff1d(range(start_frame, end_frame + 1), frame_idxs)

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
        attributes: Iterable[str]
        | None = (
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
        multi_index = pd.MultiIndex.from_tuples(df.columns.swaplevel(0, 1).swaplevel(1, 2))
        df.columns = pd.MultiIndex.from_tuples(multi_index)
        df.rename_axis(["TeamID", "PlayerID", "Attributes"], axis=1, inplace=True)
        df.sort_index(axis=1, inplace=True)

        return BBoxDataFrame(df)


