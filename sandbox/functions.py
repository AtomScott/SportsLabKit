from curses import meta
import datetime
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysrt
from omegaconf import OmegaConf

sys.path.append("../../")

from soccertrack.utils import (
    ImageIterator,
    MovieIterator,
    cv2pil,
    logger,
    make_video,
    tqdm,
)


def cut_video_file(
    video_file_name: str, start_time: int, end_time: int, save_path: str
) -> None:

    """Cut a video from start_time to end_time.
    Args:
        video_file_name (str) : Path to the video file to cut.
        start_time (int) : Start time of the video.
        duration (int) : Duration from start_time.
        save_path (str) : Path to save the video.
    """

    out_options = {"vcodec": "libx264", "crf": 23, "preset": "slow"}
    save_path.parents[0].mkdir(exist_ok=True, parents=True)
    ffmpeg.input(str(video_file_name), ss=start_time, t=end_time - start_time).output(
        str(save_path), **out_options
    ).run(overwrite_output=True)


# def cut_gps_file(gps_file_name: str, start_time: int, end_time: int, save_dir: str) -> None:

#     """Cut a gps file from start_time to end_time.

#     Args:
#         gps_file_name (str) : Path to the gps file to cut.
#         start_time (int) : Start time of the gps file.
#         end_time (int) : End time of the gps file.
#         save_path (str) : Path to save the gps file.
#     """
#     pass


def last_row_append(df_list: List[pd.DataFrame], df_list_idx: int) -> None:
    """dfの最後の行を追加（各動画の最後のフレームに該当するSRTの値が存在しないため）
    Args:
    df_list(List[pd.DataFrame]): DataFrame list.
    df_list_idx : int
            対象とするdfのリストのインデックス
    Returns
        """
    timestamp_last = df_list[df_list_idx].iloc[-1]["timestamp"]  # 最後の行のtimestamp
    timestamp_next = df_list[df_list_idx + 1].iloc[0][
        "timestamp"
    ]  # 次のdfの最初の行のtimestamp
    dur_last = df_list[df_list_idx].iloc[-1]["duration(ms)"]  # 最後の行のduration
    date_dt_last = datetime.datetime.strptime(timestamp_last, "%Y-%m-%d %H:%M:%S.%f")
    date_dt_next = datetime.datetime.strptime(timestamp_next, "%Y-%m-%d %H:%M:%S.%f")
    date_dt_last = date_dt_last + datetime.timedelta(
        milliseconds=int(dur_last)
    )  # 時間を足す
    date_last_str = date_dt_last.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    dur_last = int((date_dt_next - date_dt_last).total_seconds() * 1000)

    print(
        date_last_str,
        df_list[df_list_idx].iloc[-1]["lat"],
        df_list[df_list_idx].iloc[-1]["lon"],
        df_list[df_list_idx].iloc[-1]["alt"],
        dur_last,
    )

    df_list[df_list_idx] = df_list[df_list_idx].append(
        {
            "timestamp": date_last_str,
            "lat": df_list[df_list_idx].iloc[-1]["lat"],
            "lon": df_list[df_list_idx].iloc[-1]["lon"],
            "alt": df_list[df_list_idx].iloc[-1]["alt"],
            "duration(ms)": dur_last,
        },
        ignore_index=True,
    )


def infer_metadata_from_filename(filename):
    # TODO

    filename = Path(filename)
    basename = filename.name

    try:
        teamid = int(basename.split("_")[1])
        playerid = int(basename.split("_")[2].split(".")[0])
    except IndexError:
        teamid = 0
        playerid = 0
    metadata = {
        "teamid": teamid,
        "playerid": playerid,
    }
    return metadata


def load_gpsports(
    file_name: str, playerid: Optional[int] = None, teamid: Optional[int] = None
) -> pd.DataFrame:
    """
    Args:
        file_name(str): Path to gpsports file.

    Returns:
        gpsports_dataframe(pd.DataFrame): DataFrame of gpsports file.
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
    gpsports_dataframe = pd.DataFrame(raw_df.values, index=raw_df.index, columns=idx)
    gpsports_dataframe.index = gpsports_dataframe.index.map(
        lambda x: x.time()
    )  # remove date
    return gpsports_dataframe


def load_statsports(
    file_name: str, playerid: Optional[int] = None, teamid: Optional[int] = None
) -> pd.DataFrame:
    """
    Args:
        file_name(str): Path to statsports file.

    Returns:
        statsports_dataframe(pd.DataFrame): DataFrame of statsports file.

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
    statsports_dataframe = pd.DataFrame(raw_df.values, index=raw_df.index, columns=idx)
    statsports_dataframe.index = statsports_dataframe.index.map(lambda x: x.time())

    return statsports_dataframe


def load_gps(df_list: List[pd.DataFrame]) -> pd.DataFrame:
    # load_gpsports: pd.DataFrame, load_statsports: pd.DataFrame,
    """GPSPORTSとSTATSPORTSのファイルのマージ

    Args:
        gpsports_dataframe(pd.DataFrame): DataFrame of gpsports file.
        statsports_dataframe(pd.DataFrame): DataFrame of statsports file.

    Returns:
        merged_dataframe(pd.DataFrame): DataFrame of merged gpsports and statsports.
    """

    merged_dataframe = df_list[0].join(df_list[1 : len(df_list)])
    merged_dataframe = merged_dataframe.sort_index().interpolate()
    # merged_dataframe.index = merged_dataframe.set_index('Time')

    return merged_dataframe

    # except IndexError:
    #     merged_dataframe = funcs[0]()
    #     return merged_dataframe


def load_gps_from_yaml(yaml_path: str) -> pd.DataFrame:
    """
    Args:
        yaml_path(str): Path to yaml file.

    Returns:
        merged_dataframe(pd.DataFrame): DataFrame of merged gpsports and statsports.
    """

    assert Path.exists(yaml_path)
    cfg = OmegaConf.load(yaml_path)
    df_list = []
    for i in range(len(cfg["Device"])):
        label, gps_dir = cfg["Device"][i].values()
        if label == "STATSPORTS":
            file_name_list = sorted(list(Path(gps_dir).glob("*.csv")), reverse=False)
            for file_name in file_name_list:
                print(file_name)
                gps_df = load_statsports(str(file_name))
                df_list.append(gps_df)

        elif label == "GPSPORTS":
            file_name_list = sorted(list(Path(gps_dir).glob("*.xlsx")), reverse=False)
            for file_name in file_name_list:
                gps_df = load_gpsports(str(file_name))
                print(file_name)
                df_list.append(gps_df)

    merged_dataframe = load_gps(df_list)
    return merged_dataframe


def get_split_time(
    drone_log_dir: str,
    start_frame: int,
    end_frame: int,
):  # -> Tuple(time, int):
    """srtファイル(ドローン映像の飛行ログ)から分割に使用するタイムスタンプ(start_time, end_time)を取得する
    Args:
        drone_log_dir(str): Path to drone log directory.
        start_frame(int): Start frame of split.
        end_frame(int): End frame of split.

    Returns:
        start_time(int): Start frame of split.
        start_time(int): End frame of split.

    Notes:
        (6/2) Not readableだから修正する
    """
    drone_log_dir = sorted(list(Path(drone_log_dir).glob("*.SRT")), reverse=False)
    df_list = []
    for file_name in drone_log_dir:
        # print(file_name)
        srt_obj = pysrt.open(str(file_name))
        SRT_results = []
        for i in tqdm(range(len(srt_obj))):
            text = srt_obj[i].text_without_tags
            # timestamp
            timestamp = text.split("\n")[1]
            # latitude
            lat_search = r"\[latitude: (.*)\]"
            r_lat = re.findall(lat_search, text)
            lat = r_lat[0].split("]")[0]
            # longitude
            lon_search = r"\[longitude: (.*)\]"
            r_lon = re.findall(lon_search, text)
            lon = r_lon[0].split("]")[0]
            # altitude
            alt_search = r"\[rel_alt: (.*)\]"
            r_alt = re.findall(alt_search, text)
            alt = r_alt[0].split("]")[0].split(" ")[0]
            # duration
            dur = int(srt_obj[i].duration.milliseconds)
            SRT_result = [timestamp, lat, lon, alt, dur]
            SRT_results.append(SRT_result)

        df = pd.DataFrame(
            SRT_results, columns=["Time", "Lat", "Lon", "Alt", "duration(ms)"]
        )
        df_list.append(df)

    for i in range(0, len(df_list), 1):
        # 最後の行に追加
        try:
            last_row_append(df_list_idx=i)
        except IndexError:
            print("IndexError")
            break

    drone_df_match = pd.concat(df_list, ignore_index=True)
    start_time = drone_df_match.iloc[start_frame]
    end_time = drone_df_match.iloc[end_frame]
    return start_time, end_time


def cut_gps_file(
    gps_file_name: str, start_time: int, end_time: int, save_dir: str
) -> None:

    """Cut a gps file from start_time to end_time.

    Args:
        gps_file_name (str) : Path to the gps file to cut.
        start_time (int) : Start time of split.
        end_time (int) : End time of split.
        save_path (str) : Path to save the gps file.
    """
    pass


def visualization_gps(gps_file_name: str, save_path: str) -> None:

    """Visualize the gps file.

    Args:
        gps_file_name (str) : Path to the gps file to visualize. #整形したGPSデータ(csv)を指定
        save_path (str) : Path to save the gps file
    """

    pass


def visualization_annotations(annotations_file_name: str, save_path: str) -> None:

    """Visualize the annotations file.

    Args:
        annotations_file_name (str) : Path to the annotations file to visualize.
        save_path (str) : Path to save the annotations file.
    """
    pass


def upload2s3(
    integration_key: str, bucket_name: str, file_name: str
) -> bool:  # Not sure how to integrate with S3, but probably need to fill in some kind of key or bucket name

    """Upload a file to S3.

    Args:
        integration_key (str) : Integration key for the S3 bucket.
        bucket_name (str) : Name of the S3 bucket.
        file_name (str) : Name of the file to upload.

    Returns:
        bool (bool) : True if the upload was successful, False otherwise.

    """
    pass


def download_from_s3(
    integration_key: str, bucket_name: str, download_dir: str, save_path: str
) -> bool:

    """Download a file from S3.

    Args:
        integration_key (str) : Integration key for the S3 bucket.
        bucket_name (str) : Name of the S3 bucket.
        download_dir (str) :Path of the directory to download from S3.
        save_path (str) : Path to save the file.

    Returns:
        bool (bool) : True if the download was successful, False otherwise.

    Note:
        Save S3 directory as a zip file
    """
    pass


def upload_annotation2labelbox(
    annotations_file_name: str, labelbox_api_key: str, labelbox_project_id: str
) -> bool:  # Probably some kind of Labelbox access key is needed.

    """Upload annotations to Labelbox.

    Args:
        annotations_file_name (str) : Path to the annotations file to upload.
        labelbox_api_key (str) : Labelbox API key.
        labelbox_project_id (str) : Labelbox project ID.

    Returns:
        bool (bool) : True if the download was successful, False otherwise.
    """
    pass


def upload_video2labelbox(
    video_file_name: str, labelbox_api_key: str, labelbox_project_id: str
) -> bool:  # Probably some kind of Labelbox access key is needed.

    """Upload video to Labelbox.

    Args:
        video_file_name (str) : Path to the video file to upload.
        labelbox_api_key (str) : Labelbox API key.
        labelbox_project_id (str) : Labelbox project ID.
    Returns:
        bool (bool) : True if the upload was successful, False otherwise.
    """
    pass


def create_annotation_df_from_s3(
    integration_key: str,
    bucket_name: str,
    root_dir: str,
    dir_name_list: List[str],
    save_path: str,
) -> None:

    """Create a dataframe(csv file) from the annotations file.

    Args:
        integration_key (str) : Integration key for the S3 bucket.
        bucket_name (str) : Name of the S3 bucket.
        root_dir (str) : Root directory of the S3 bucket.
        dir_name_list (list[str]) :List of data types to be stored in the df columns. The element of each list contains the directory name.
        save_path (str) : Path to save the csv_file.
    """
    pass
