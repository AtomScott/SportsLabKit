from itertools import zip_longest
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf
from typing import Optional

from soccertrack import GPSDataFrame, BBoxDataFrame
def infer_metadata_from_filename(filename):
    # TODO

    filename = Path(filename)
    basename = filename.name

    try:
        teamid = int(basename.split("_")[1])
        playerid = int(basename.split("_")[2].split(".")[0])
    except (IndexError, ValueError): # TODO: Make this exception more specific
        teamid = 0
        playerid = 0
    metadata = {
        "teamid": teamid,
        "playerid": playerid,
    }
    return metadata

def load_gpsports(
    file_name: str, playerid: Optional[int] = None, teamid: Optional[int] = None
) -> GPSDataFrame:
    """
    Args:
        file_name(str): Path to gpsports file.

    Returns:
        gpsports_dataframe(GPSDataFrame): DataFrame of gpsports file.
    Notes:
        出力するDataFrameの列は以下の通り
        Time(datetime): GPS(GPSPORTS)のタイムスタンプ
        Lat(float): GPSの緯度
        Lon(float): GPSの経度
    """
    # read_gpsdata
    raw_df = pd.read_excel(
        file_name,
        skiprows=7,
        usecols=["Time", "Latitude", "Longitude"],
        index_col="Time",
    ).rename(columns={"Latitude": "Lat", "Longitude": "Lon"})
    # get multicolumn

    metadata = infer_metadata_from_filename(file_name)
    teamid = teamid if teamid is not None else metadata.get("teamid")
    playerid = playerid if playerid is not None else metadata.get("playerid")

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
    file_name: str, playerid: Optional[int] = None, teamid: Optional[int] = None
) -> GPSDataFrame:
    """
    Args:
        file_name(str): Path to statsports file.

    Returns:
        statsports_dataframe(GPSDataFrame): DataFrame of statsports file.

    Notes:
        出力するDataFrameの列は以下の通り
        Time(datetime): GPS(STATSPORTS)のタイムスタンプ
        Lat(float): GPSの緯度
        Lon(float): GPSの経度
    """
    raw_df = (
        pd.read_csv(file_name)
        .iloc[:, [1, 3, 4]]
        .set_axis(["Time", "Lat", "Lon"], axis="columns")
        .reset_index(drop=True)
    )
    raw_df["Time"] = pd.to_datetime(raw_df["Time"])
    raw_df.set_index("Time", inplace=True)

    metadata = infer_metadata_from_filename(file_name)
    teamid = teamid if teamid is not None else metadata.get("teamid")
    playerid = playerid if playerid is not None else metadata.get("playerid")

    idx = pd.MultiIndex.from_arrays(
        [[int(teamid)] * 2, [int(playerid)] * 2, list(raw_df.columns)],
    )

    # change multicolumn
    statsports_dataframe = GPSDataFrame(raw_df.values, index=raw_df.index, columns=idx)
    statsports_dataframe.index = statsports_dataframe.index.map(lambda x: x.time())

    return statsports_dataframe


def infer_gps_format(file_name):
    file_name = str(file_name)
    # TODO: 拡張子で判定する方法はザルすぎる
    if file_name.endswith(".xlsx"):
        return "gpsports"
    elif file_name.endswith(".csv"):
        return "statsports"
    else:
        raise ValueError("Could not infer file format")


def get_gps_loader(format):
    format = format.lower()
    if format == "gpsports":
        return load_gpsports
    if format == "statsports":
        return load_statsports
    raise ValueError(f"Unknown format {format}")


def load_gps(
    file_names: list[str], playerids: list[int] = [], teamids: list[int] = []
) -> GPSDataFrame:
    # load_gpsports: GPSDataFrame, load_statsports: GPSDataFrame,
    """GPSPORTSとSTATSPORTSのファイルのマージ # TODO: 修正

    Args:
        gpsports_dataframe(GPSDataFrame): DataFrame of gpsports file.
        statsports_dataframe(GPSDataFrame): DataFrame of statsports file.

    Returns:
        merged_dataframe(GPSDataFrame): DataFrame of merged gpsports and statsports.
    """

    if not isinstance(file_names, (list, tuple)):
        file_names = [file_names]

    playerid = 0  # TODO: 付与ロジックを書く
    teamid = None  # TODO 付与ロジックを書く

    if not isinstance(playerids, (list, tuple)):
        playerids = [playerids]

    if not isinstance(teamids, (list, tuple)):
        teamids = [teamids]

    df_list = []
    for i, (file_name, playerid, teamid) in enumerate(
        zip_longest(file_names, playerids, teamids)
    ):
        playerid = playerid if playerid is not None else i
        gps_format = infer_gps_format(file_name)
        dataframe = get_gps_loader(gps_format)(file_name, playerid, teamid)
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
    """
    Args:
        yaml_path(str): Path to yaml file.

    Returns:
        merged_dataframe(GPSDataFrame): DataFrame of merged gpsports and statsports.
    """

    cfg = OmegaConf.load(yaml_path)
    df_list = []
    playerids, teamids, filepaths = [], [], []
    for device in cfg.devices:
        playerids.append(device.playerid)
        teamids.append(device.teamid)
        filepaths.append(Path(device.filepath))

    return load_gps(filepaths, playerids, teamids)



# def infer_metadata_from_filename(filename):
#     # TODO

#     filename = Path(filename)
#     basename = filename.name

#     try:
#         teamid = int(basename.split("_")[1])
#         playerid = int(basename.split("_")[2].split(".")[0])
#     except (IndexError, ValueError): # TODO: Make this exception more specific
#         teamid = 0
#         playerid = 0
#     metadata = {
#         "teamid": teamid,
#         "playerid": playerid,
#     }
#     return metadata

def load_labelbox(
    file_name: str
) -> GPSDataFrame:
    """
    Args:
        file_name(str): Path to gpsports file.

    Returns:
        gpsports_dataframe(GPSDataFrame): DataFrame of gpsports file.
    Notes:
        出力するDataFrameの列は以下の通り
        Time(datetime): GPS(GPSPORTS)のタイムスタンプ
        Lat(float): GPSの緯度
        Lon(float): GPSの経度
    """
    # read_gpsdata
    df = pd.read_json(file_name, lines=True).explode('objects')
    objects_df = df["objects"].apply(pd.Series)
    bbox_df = objects_df["bbox"].apply(pd.Series)

    df = pd.concat([
        df[['frameNumber']],
        objects_df.title.str.split('_', expand=True),
        bbox_df[['left', 'top', 'width', 'height']],
        ], axis=1)

    df.columns = ['frame', 'teamid', 'playerid', 'bb_left', 'bb_top', 'bb_width', 'bb_height']
    df.set_index('frame', inplace=True)

    groups = df.groupby('playerid', dropna=False)

    df_list = []
    for playerid, group in groups:
        teamid = group.teamid.iloc[0]
        bbox_cols = ['bb_left', 'bb_top', 'bb_width', 'bb_height']
        
        if teamid.lower() =='sports ball':
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

def load_mot(
    file_name
) -> GPSDataFrame:
    """
    Args:
        file_name(str): Path to statsports file.

    Returns:
        statsports_dataframe(GPSDataFrame): DataFrame of statsports file.

    Notes:
        出力するDataFrameの列は以下の通り
        Time(datetime): GPS(STATSPORTS)のタイムスタンプ
        Lat(float): GPSの緯度
        Lon(float): GPSの経度
    """
    groups = pd.read_csv(file_name, usecols=[0, 1, 2, 3, 4, 5], index_col=0).groupby('id')

    teamid = 0
    # playerid = 0

    df_list = []
    for playerid, group in groups:
        group['conf'] = 1.0
        group['class_id'] = int(0) #classid of person
        if playerid == 23:
            group['class_id'] = int(32) #classid of person
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


def infer_bbox_format(file_name):
    file_name = str(file_name)
    # TODO: 拡張子で判定する方法はザルすぎる
    if file_name.endswith(".csv"):
        return "mot"
    elif file_name.endswith(".ndjson"):
        return "labelbox"
    else:
        raise ValueError("Could not infer file format")


def get_bbox_loader(format):
    format = format.lower()
    if format == "mot":
        return load_mot
    if format == "labelbox":
        return load_labelbox
    raise ValueError(f"Unknown format {format}")


def load_bboxes(
    file_name
) -> BBoxDataFrame:
    # load_gpsports: BBoxDataFrame, load_statsports: GPSDataFrame,
    """GPSPORTSとSTATSPORTSのファイルのマージ # TODO: 修正

    Args:
        gpsports_dataframe(BBoxDataFrame): DataFrame of gpsports file.
        statsports_dataframe(BBoxDataFrame): DataFrame of statsports file.

    Returns:
        merged_dataframe(BBoxDataFrame): DataFrame of merged gpsports and statsports.
    """

    gps_format = infer_bbox_format(file_name)
    return  get_bbox_loader(gps_format)(file_name)


def load_bboxes_from_yaml(yaml_path: str) -> BBoxDataFrame:
    """
    Args:
        yaml_path(str): Path to yaml file.

    Returns:
        merged_dataframe(BBoxDataFrame): 
    """

    cfg = OmegaConf.load(yaml_path)
    df_list = []
    playerids, teamids, filepaths = [], [], []
    for device in cfg.devices:
        playerids.append(device.playerid)
        teamids.append(device.teamid)
        filepaths.append(Path(device.filepath))

    return load_bboxes(filepaths, playerids, teamids)


# %%
