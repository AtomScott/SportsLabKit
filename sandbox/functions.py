from curses import meta
import datetime
import os
import re
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple
from itertools import zip_longest
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io

import cv2 as cv
import ffmpeg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysrt
from omegaconf import OmegaConf
import dateutil.parser
from mplsoccer import Pitch
from xml.etree import ElementTree
from ast import literal_eval

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


def load_gps(file_names: list[str], playerids: list[int] = [], teamids:list[int] = []) -> pd.DataFrame:
    # load_gpsports: pd.DataFrame, load_statsports: pd.DataFrame,
    """GPSPORTSとSTATSPORTSのファイルのマージ # TODO: 修正

    Args:
        gpsports_dataframe(pd.DataFrame): DataFrame of gpsports file.
        statsports_dataframe(pd.DataFrame): DataFrame of statsports file.

    Returns:
        merged_dataframe(pd.DataFrame): DataFrame of merged gpsports and statsports.
    """

    if not isinstance(file_names, (list, tuple)):
        file_names = [file_names]

    playerid = 0 # TODO: 付与ロジックを書く
    teamid = None # TODO 付与ロジックを書く

    if not isinstance(playerids, (list, tuple)):
        playerids = [playerids]

    if not isinstance(teamids, (list, tuple)):
        teamids = [teamids]

    df_list = []
    for i, (file_name, playerid, teamid) in enumerate(zip_longest(file_names, playerids, teamids)):
        playerid = playerid if playerid is not None else i
        gps_format = infer_gps_format(file_name)
        dataframe = get_gps_loader(gps_format)(file_name, playerid, teamid)
        df_list.append(dataframe)

        playerid += 1 #TODO: これではyamlから読み込むことができない

    merged_dataframe = df_list[0].join(df_list[1 : len(df_list)]) # これができるのは知らなかった
    merged_dataframe = merged_dataframe.sort_index().interpolate() # 暗黙的にinterpolateするのが正解なのか？

    merged_dataframe = df_list[0].join(df_list[1 : len(df_list)])
    merged_dataframe = merged_dataframe.sort_index().interpolate()
    return merged_dataframe


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
    playerids, teamids, filepaths = [], [], []
    for device in cfg.devices:
        playerids.append(device.playerid)
        teamids.append(device.teamid)
        filepaths.append(Path(device.filepath))
        
    return load_gps(filepaths, playerids, teamids)


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
            print("IndexError") #<- 例外処理を消すor処理内容をより明示的に書く
            break

    drone_df_match = pd.concat(df_list, ignore_index=True)
    start_time = datetime.datetime.strptime(drone_df_match.iloc[start_frame]['timestamp'], "%Y-%m-%d %H:%M:%S.%f").time()
    end_time = datetime.datetime.strptime(drone_df_match.iloc[end_frame]['timestamp'], "%Y-%m-%d %H:%M:%S.%f").time()
    return start_time, end_time

def auto_string_parser(value:str):
    # automatically parse values to correct type
    if value.isdigit():
        return int(value)
    elif value.replace('.', '', 1).isdigit():
        return float(value)
    elif value.lower() == 'true':
        return True
    elif value.lower() == 'false':
        return False
    elif value.lower() == 'nan':
        return np.nan
    elif value.lower() == 'inf':
        return np.inf
    elif value.lower() == '-inf':
        return -np.inf
    else:
        try:
            return literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        try:
            return dateutil.parser.parse(value)
        except (ValueError, TypeError):
            pass
    return value


def save_dataframe(df, path_or_buf):

    if df.attrs:
        # write dataframe attributes to the csv file
        with open(path_or_buf, 'w') as f:
            for k, v in df.attrs.items():
                f.write(f'#{k}:{v}\n')

    df.to_csv(path_or_buf, mode='a')


def load_dataframe(path_or_buf):
    attrs = {}
    with open(path_or_buf, 'r') as f:
        for line in f:
            if line.startswith('#'):
                k, v = line[1:].strip().split(':')
                attrs[k] = auto_string_parser(v)
            else:
                break

    skiprows = len(attrs)
    df = pd.read_csv(path_or_buf, header=[0,1,2], index_col=0, skiprows=skiprows)
    df.attrs = attrs
    return df

def cut_gps_file(
    gps_file_name: str, start_time: int, end_time: int, save_dir: str
) -> None:

    """Cut a gps file from start_time to end_time.

    Args:
        gps_file_name (str) : Path to the gps file to cut.
        start_time (int) : Start time of split.
        end_time (int) : End time of split.
        save_dir (str) : Directory's path to save the cut gps file.
    """
    pass

def get_homography_from_kml(kml_file_name: str) -> np.ndarray:

    """Get homography matrix.

    Args:
        kml_file_name(str): Path to the kml file to get homography matrix.

    Returns:
        H(np.ndarray): Homography matrix.
    """

    # kmlfile = '/home/atom/SoccerTrack/notebooks/GPS_data/2022_02_20/pitch_annotation_tsukuba.kml'
    tree = ElementTree.parse(kml_file_name)
    src = []
    dst = []
    ns = "{http://www.opengis.net/kml/2.2}"
    placemarks = tree.findall(".//%sPlacemark" % ns)
    for placemark in placemarks:
        label = placemark.find("./%sname" % ns).text
        coordinates = placemark.find("./%sPoint/%scoordinates" % (ns,ns)).text.split(',')
        lon, lat = coordinates[0], coordinates[1]
        dst.append(eval(label))
        src.append(eval(lon+','+ lat))
    source_keypoints = np.asarray(src)
    target_keypoints = np.asarray(dst)

    H, *_ = cv.findHomography(source_keypoints, target_keypoints, cv.RANSAC, 5.0)
    return H

def get_Transforms(df: pd.DataFrame, H: np.ndarray) -> np.ndarray:
    """Get Transforms from GPS data.

    Args:
        df(pd.DataFrame): GPS data.
        H(np.ndarray): Homography matrix.

    Returns:
        xsys(np.ndarray): Transformed GPS information.
    """
    _xsys = np.expand_dims(df.values, axis=0).astype('float64')
    xsys = cv.perspectiveTransform(_xsys, H).T.squeeze()

    # transpose to get (xs) and (ys)
    return xsys

def fig2img(fig, ax, dpi=180) -> np.ndarray:
    """Convert a figure to a numpy array.

    Args:
        fig(matplotlib.figure.Figure): Figure to convert.
        ax(matplotlib.axes.Axes): Axes to convert.
        dpi(int): DPI of the figure.

    Returns:
        img(numpy.ndarray): Converted figure.
    """
    ax.invert_yaxis()
    ax.axis('off')
    canvas = FigureCanvas(fig)
    canvas.draw()       # draw the canvas, cache the renderer
    buf = io.BytesIO()  # インメモリのバイナリストリームを作成
    fig.savefig(buf, format="png", dpi=dpi)  # matplotlibから出力される画像のバイナリデータをメモリに格納する.
    buf.seek(0)  # ストリーム位置を先頭に戻る
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)  # メモリからバイナリデータを読み込み, numpy array 形式に変換
    img = cv.imdecode(img_arr, 1)  # 画像のバイナリデータを復元する
    return img

def visualization_gps(kml_file_name: str, gps_file_name: str, save_path: str) -> None:
    """Visualize the gps file.
    Args:
        kml_file_name(str): Path to the kml file to get homography matrix. ピッチのキーポイントの座標(kml)を指定
        gps_file_name (str) : Path to the gps file to visualize. #整形したGPSデータ(csv)を指定
        save_path (str) : Path to save the gps file
    """
    #図形のサイズ指定
    team0_color = np.array((255, 255, 255)) / 255
    team1_color = np.array((48, 188, 212))/255
    circle_r = 10

    H = get_homography_from_kml(str(kml_file_name)) #get homograpy matrix
    gps_df = load_dataframe(str(gps_file_name)).reset_index(inplace=False) #args

    # Plot trajectory
    pitch = Pitch(
        pitch_color="black",
        line_color=(0.3, 0.3, 0.3),
        pitch_type="custom",
        pitch_length=105,
        pitch_width=68,
        label=False
    )

    ax = pitch.draw()
    ax.invert_xaxis()
    for i in tqdm(range(len(gps_df))):
        test_df = gps_df[i : i+1]
        df_list_frame = []
        id = 0
        for k in [0,1]:
            for n in range(0, 22, 1):
                teamid = k
                playerid = n
                try:
                    split = test_df[f'{teamid}'][f'{playerid}'][['Lon', 'Lat']]
                except KeyError:
                    continue
                try:
                    xs, ys = xsys = get_Transforms(split, H)
                except ValueError:
                    print(f'{teamid}_{playerid}_ValueError')
                    continue
                # print(xs, ys)
                w, h = 1, 1
                df_list_frame.append([id, xs, ys, w, h])
                id += 1
        frame_df = pd.DataFrame(df_list_frame, columns=['id', 'x', 'y', 'w', 'h'])
        scaler = 1 + (i - len(gps_df))/len(gps_df)
        # print(scaler)
        for _, row in frame_df.iterrows():
            if row.id < 11:
                ax.scatter(row.x + row.w / 2, row.y + row.h / 2, s=circle_r * scaler, color=team0_color, alpha=1 * scaler)
            else:
                ax.scatter(row.x + row.w / 2, row.y + row.h / 2, s=circle_r * scaler, color=team1_color, alpha=1 * scaler)

    plt.savefig(save_path)
    plt.close()
    plt.cla()
    plt.clf()


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