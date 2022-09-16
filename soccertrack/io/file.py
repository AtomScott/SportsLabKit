from __future__ import annotations

import csv
import os
from ast import literal_eval
from itertools import zip_longest
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import dateutil.parser
import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from soccertrack import BBoxDataFrame, GPSDataFrame

_pathlike = Union[str, os.PathLike[str]]


def auto_string_parser(value: str) -> Any:
    """Auxiliary function to parse string values.

    Args:
        value (str): String value to parse.

    Returns:
        value (any): Parsed string value.
    """
    # automatically parse values to correct type
    if value.isdigit():
        return int(value)
    if value.replace(".", "", 1).isdigit():
        return float(value)
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.lower() == "nan":
        return np.nan
    if value.lower() == "inf":
        return np.inf
    if value.lower() == "-inf":
        return -np.inf

    try:
        return literal_eval(value)
    except (ValueError, SyntaxError):
        pass
    try:
        return dateutil.parser.parse(value)
    except (ValueError, TypeError):
        pass
    return value


def infer_metadata_from_filename(filename: _pathlike) -> Mapping[str, int]:
    """Try to infer metadata from filename.

    Args:
        filename (Union[str, bytes, os.PathLike[Any]]): Filename to infer metadata from

    Returns:
        dict[str, Union[int, str]]: Dictionary with metadata
    """

    filename = Path(filename)
    basename = filename.name

    try:
        teamid = int(basename.split("_")[1])
        playerid = int(basename.split("_")[2].split(".")[0])
    except (IndexError, ValueError):  # TODO: Make this exception more specific
        teamid = 0
        playerid = 0
    metadata = {
        "teamid": teamid,
        "playerid": playerid,
    }
    return metadata


def load_gpsports(
    filename: _pathlike,
    playerid: Optional[int] = None,
    teamid: Optional[int] = None,
) -> GPSDataFrame:
    """Load GPSDataFrame from GPSPORTS file.

    Args:
        filename(str): Path to gpsports file.

    Returns:
        gpsports_dataframe(GPSDataFrame): DataFrame of gpsports file.
    """
    # read_gpsdata
    raw_df = pd.read_excel(
        filename,
        skiprows=7,
        usecols=["Time", "Latitude", "Longitude"],
        index_col="Time",
    ).rename(columns={"Latitude": "Lat", "Longitude": "Lon"})

    # get multicolumn
    metadata = infer_metadata_from_filename(filename)
    teamid = teamid if teamid is not None else metadata["teamid"]
    playerid = playerid if playerid is not None else metadata["playerid"]

    idx = pd.MultiIndex.from_arrays(
        [[int(teamid)] * 2, [int(playerid)] * 2, list(raw_df.columns)],
    )

    # Change single column to multi-column
    gpsports_dataframe = GPSDataFrame(raw_df.values, index=raw_df.index, columns=idx)
    gpsports_dataframe.index = gpsports_dataframe.index.map(
        lambda x: x.time()
    )  # remove date
    return gpsports_dataframe


def load_statsports(
    filename: _pathlike,
    playerid: Optional[int] = None,
    teamid: Optional[int] = None,
) -> GPSDataFrame:
    """Load GPSDataFrame from STATSPORTS file.

    Args:
        filename(str): Path to statsports file.

    Returns:
        statsports_dataframe(GPSDataFrame): DataFrame of statsports file.
    """
    raw_df = (
        pd.read_csv(filename)
        .iloc[:, [1, 3, 4]]
        .set_axis(["Time", "Lat", "Lon"], axis="columns")
        .reset_index(drop=True)
    )
    raw_df["Time"] = pd.to_datetime(raw_df["Time"])
    raw_df.set_index("Time", inplace=True)

    metadata = infer_metadata_from_filename(filename)
    teamid = teamid if teamid is not None else metadata["teamid"]
    playerid = playerid if playerid is not None else metadata["playerid"]

    idx = pd.MultiIndex.from_arrays(
        [[int(teamid)] * 2, [int(playerid)] * 2, list(raw_df.columns)],
    )

    # change multicolumn
    statsports_dataframe = GPSDataFrame(raw_df.values, index=raw_df.index, columns=idx)
    statsports_dataframe.index = statsports_dataframe.index.map(lambda x: x.time())

    return statsports_dataframe


def infer_gps_format(filename: _pathlike) -> str:
    """Try to infer GPS format from filename.

    Args:
        filename (Union[str, bytes, os.PathLike[Any]]): Filename to infer format from.
    """

    filename = str(filename)

    if filename.endswith(".xlsx"):
        return "gpsports"
    if filename.endswith(".csv"):
        return "statsports"
    raise ValueError("Could not infer file format")


def get_gps_loader(
    format: str,
) -> Callable[[_pathlike, int, int], GPSDataFrame]:
    """Get GPS loader function for a given format.

    Args:
        format (str): GPS format.

    Returns:
        Callable[[_pathlike, int, int], GPSDataFrame]: GPS loader function.
    """
    format = format.lower()
    if format == "gpsports":
        return load_gpsports
    if format == "statsports":
        return load_statsports
    raise ValueError(f"Unknown format {format}")


def load_gps(
    filenames: Union[
        Sequence[
            _pathlike,
        ],
        _pathlike,
    ],
    playerids: Union[Sequence[int], int] = (),
    teamids: Union[Sequence[int], int] = (),
) -> GPSDataFrame:
    """Load GPS data from multiple files.

    Args:
        gpsports_dataframe(GPSDataFrame): DataFrame of gpsports file.
        statsports_dataframe(GPSDataFrame): DataFrame of statsports file.

    Returns:
        merged_dataframe(GPSDataFrame): DataFrame of merged gpsports and statsports.
    """

    if not isinstance(filenames, Sequence):
        filenames = [filenames]

    playerid = 0  # TODO: 付与ロジックを書く
    teamid = None  # TODO 付与ロジックを書く

    if not isinstance(playerids, Sequence):
        playerids = [playerids]

    if not isinstance(teamids, Sequence):
        teamids = [teamids]

    df_list = []
    for i, (filename, playerid, teamid) in enumerate(
        zip_longest(filenames, playerids, teamids)
    ):
        playerid = playerid if playerid is not None else i
        gps_format = infer_gps_format(filename)
        dataframe = get_gps_loader(gps_format)(filename, playerid, teamid)
        df_list.append(dataframe)

        playerid += 1  # TODO: これではyamlから読み込むことができない

    merged_dataframe = df_list[0].join(df_list[1 : len(df_list)])  # これができるのは知らなかった
    merged_dataframe = (
        merged_dataframe.sort_index().interpolate()
    )  # 暗黙的にinterpolateするのが正解なのか？

    merged_dataframe = df_list[0].join(df_list[1 : len(df_list)])
    merged_dataframe = merged_dataframe.sort_index().interpolate()

    # 座標変換する？
    return merged_dataframe


def load_gps_from_yaml(yaml_path: str) -> GPSDataFrame:
    """Load GPS data from a YAML file.

    Args:
        yaml_path(str): Path to yaml file.

    Returns:
        merged_dataframe(GPSDataFrame): DataFrame of merged gpsports and statsports.
    """

    cfg = OmegaConf.load(yaml_path)
    playerids, teamids, filepaths = [], [], []
    for device in cfg.devices:
        playerids.append(device.playerid)
        teamids.append(device.teamid)
        filepaths.append(Path(device.filepath))

    return load_gps(filepaths, playerids, teamids)


def load_labelbox(filename: _pathlike) -> GPSDataFrame:
    """Load labelbox format file to GPSDataFrame.

    Args:
        filename(str): Path to gpsports file.

    Returns:
        gpsports_dataframe(GPSDataFrame): DataFrame of gpsports file.

    Notes:
        出力するDataFrameの列は以下の通り
        Time(datetime): GPS(GPSPORTS)のタイムスタンプ
        Lat(float): GPSの緯度
        Lon(float): GPSの経度
    """
    # read_gpsdata
    df = pd.read_json(filename, lines=True).explode("objects")
    objects_df = df["objects"].apply(pd.Series)
    bbox_df = objects_df["bbox"].apply(pd.Series)

    df = pd.concat(
        [
            df[["frameNumber"]],
            objects_df.title.str.split("_", expand=True),
            bbox_df[["left", "top", "width", "height"]],
        ],
        axis=1,
    )

    df.columns = [
        "frame",
        "teamid",
        "playerid",
        "bb_left",
        "bb_top",
        "bb_width",
        "bb_height",
    ]
    df.set_index("frame", inplace=True)

    groups = df.groupby("playerid", dropna=False)

    df_list = []
    for playerid, group in groups:
        teamid = group.teamid.iloc[0]
        bbox_cols = ["bb_left", "bb_top", "bb_width", "bb_height"]

        if teamid.lower() == "sports ball":
            teamid = 3
            playerid = 0

        idx = pd.MultiIndex.from_arrays(
            [[int(float(teamid))] * 4, [int(float(playerid))] * 4, bbox_cols],
        )

        bbox_df = BBoxDataFrame(group[bbox_cols].values, index=group.index, columns=idx)
        df_list.append(bbox_df)

    merged_dataframe = df_list[0].join(df_list[1 : len(df_list)])  # これができるのは知らなかった
    merged_dataframe = (
        merged_dataframe.sort_index().interpolate()
    )  # 暗黙的にinterpolateするのが正解なのか？

    merged_dataframe = df_list[0].join(df_list[1 : len(df_list)])
    merged_dataframe = merged_dataframe.sort_index().interpolate()

    return merged_dataframe


def load_mot(filename: _pathlike) -> GPSDataFrame:
    """Load MOT format file to GPSDataFrame.

    Args:
        filename(str): Path to statsports file.

    Returns:
        statsports_dataframe(GPSDataFrame): DataFrame of statsports file.

    Notes:
        出力するDataFrameの列は以下の通り
        Time(datetime): GPS(STATSPORTS)のタイムスタンプ
        Lat(float): GPSの緯度
        Lon(float): GPSの経度
    """
    groups = pd.read_csv(filename, usecols=[0, 1, 2, 3, 4, 5], index_col=0).groupby(
        "id"
    )

    teamid = 0
    # playerid = 0

    df_list = []
    for playerid, group in groups:
        group['conf'] = 1.0
        group['class_id'] = int(0) # TODO: classid of person
        if playerid == 23:
            group['class_id'] = int(32) # TODO: classid of ball
            teamid = 3
            playerid = 0
        elif 11 < playerid < 23:
            teamid = 1
            playerid = playerid - 11
        bbox_cols = ['bb_left', 'bb_top', 'bb_width', 'bb_height', 'conf', 'class_id']
        idx = pd.MultiIndex.from_arrays(
            [[int(teamid)] * 6, [int(playerid)] * 6, bbox_cols],
        )

        bbox_df = BBoxDataFrame(group[bbox_cols].values, index=group.index, columns=idx)
        df_list.append(bbox_df)

    merged_dataframe = df_list[0].join(df_list[1 : len(df_list)])  # これができるのは知らなかった
    merged_dataframe = (
        merged_dataframe.sort_index().interpolate()
    )  # 暗黙的にinterpolateするのが正解なのか？

    merged_dataframe = df_list[0].join(df_list[1 : len(df_list)])
    merged_dataframe = merged_dataframe.sort_index().interpolate()

    return merged_dataframe


def load_bbox(
    filename: _pathlike,
) -> pd.DataFrame:
    """Load a dataframe from a file.

    Args:
        filename (_pathlike): Path to load the dataframe.
    Returns:
        df (pd.DataFrame): Dataframe loaded from the file.
    """
    attrs = {}
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                k, v = line[1:].strip().split(":")
                attrs[k] = auto_string_parser(v)
            else:
                break

    skiprows = len(attrs)
    df = pd.read_csv(filename, header=[0, 1, 2], index_col=0, skiprows=skiprows)
    df.attrs = attrs
    return df


def is_mot(filename: _pathlike) -> bool:
    """Return True if the file is MOT format.

    Args:
        filename(_pathlike): Path to file.

    Returns:
        is_mot(bool): True if the file is MOT format.
    """
    with open(filename, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        first_line = next(reader)

    return ["frame", "id", "bb_left", "bb_top", "bb_width", "bb_height"] == first_line


def infer_bbox_format(filename: _pathlike) -> str:
    """Try to infer the format of a given bounding box file.

    Args:
        filename(_pathlike): Path to bounding box file.

    Returns:
        format(str): Inferred format of the bounding box file.
    """
    filename = str(filename)
    # TODO: 拡張子で判定する方法はザルすぎる
    if is_mot(filename):
        return "mot"
    if filename.endswith(".csv"):
        return "soccertrack_bbox"
    if filename.endswith(".ndjson"):
        return "labelbox"
    raise ValueError("Could not infer file format")


def get_bbox_loader(
    format: str,
) -> Callable[[_pathlike], BBoxDataFrame]:
    """Returns a function that loads the corresponding bbox format.

    Args:
        format(str): bbox format to load.

    Returns:
        bbox_loader(Callable[[_pathlike], BBoxDataFrame]): Function that loads the corresponding bbox format.
    """
    format = format.lower()
    if format == "mot":
        return load_mot
    if format == "labelbox":
        return load_labelbox
    if format == "soccertrack_bbox":
        return load_bbox
    raise ValueError(f"Unknown format {format}")


def load_df(filename: _pathlike) -> Union[BBoxDataFrame, GPSDataFrame]:
    """Loads either a BBoxDataFrame or a GPSDataFrame from a file.

    Args:
        filename(Uinon[str, os.PathLike[Any]]): Path to file.

    Returns:
        dataframe(Union[BBoxDataFrame, GPSDataFrame]): DataFrame of file.
    """

    gps_format = infer_bbox_format(filename)
    df = BBoxDataFrame(get_bbox_loader(gps_format)(filename))
    df.rename_axis(["TeamID", "PlayerID", "Attributes"], axis=1, inplace=True)
    return df


# def load_bboxes_from_yaml(yaml_path: _pathlike) -> BBoxDataFrame:
#     """
#     Args:
#         yaml_path(str): Path to yaml file.

#     Returns:
#         merged_dataframe(BBoxDataFrame):
#     """

#     cfg = OmegaConf.load(yaml_path)
#     df_list = []
#     playerids, teamids, filepaths = [], [], []
#     for device in cfg.devices:
#         playerids.append(device.playerid)
#         teamids.append(device.teamid)
#         filepaths.append(Path(device.filepath))

#     return load_bboxes(filepaths, playerids, teamids)
