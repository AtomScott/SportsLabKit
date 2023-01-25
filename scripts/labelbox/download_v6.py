"""Script to download data from Labelbox.
"""

import argparse
import os
from pathlib import Path

import ndjson
import requests
from dotenv import load_dotenv
from labelbox import Client

from soccertrack.dataframe import BBoxDataFrame
from soccertrack.logger import tqdm

load_dotenv()

LABELBOX_API_KEY = os.getenv("LABELBOX_API_KEY")

parser = argparse.ArgumentParser()
parser.add_argument("--project_id", help="The project ID to download from.")
parser.add_argument("--save_root", help="The path to save the downloaded data to.")
args = parser.parse_args()


def download_video(video_url: str, save_path: str):
    """Download a video from Labelbox.

    Args:
        video_url (str): The URL to download the video from.
        save_path (str): The path to save the video to.
    """
    with open(save_path, "wb") as file:
        file.write(requests.get(video_url).content)


def download_annotations(annotations_url: str, save_path: str):
    """Download annotations from Labelbox.

    Args:
        annotations_url (str): The URL to download the annotations from.
        save_path (str): The path to save the annotations to.
    """
    headers = {"Authorization": f"Bearer {LABELBOX_API_KEY}"}
    annotations = ndjson.loads(requests.get(annotations_url, headers=headers).text)
    home_team_key = "0"
    away_team_key = "1"
    ball_key = "BALL"

    d = {home_team_key: {}, away_team_key: {}, ball_key: {}}

    for annotation in annotations:
        for frame_annotation in annotation["objects"]:
            frame_number = annotation["frameNumber"]
            bbox = frame_annotation["bbox"]

            if frame_annotation["title"] == ball_key:
                team_id = ball_key
                player_id = ball_key
            else:
                team_id, player_id = frame_annotation["title"].split("_")

            if d[team_id].get(player_id) is None:
                d[team_id][player_id] = {}
            d[team_id][player_id][frame_number] = [
                bbox["left"],
                bbox["top"],
                bbox["width"],
                bbox["height"],
            ]

    bbdf = BBoxDataFrame.from_dict(d)
    bbdf.to_csv(save_path)
    return bbdf


if __name__ == "__main__":
    save_root = Path(args.save_root)

    # Set up the client.
    client = Client(api_key=LABELBOX_API_KEY)
    project = client.get_project(args.project_id)

    export_url = project.export_labels()
    exports = requests.get(export_url).json()

    for export_data in (pbar := tqdm(exports)):
        external_id = Path(export_data["External ID"])
        pbar.set_description(f"Downloading {external_id}")
        if not str(external_id).startswith("F"):
            continue

        video_url = export_data["Labeled Data"]
        annotations_url = export_data["Label"].get("frames")
        if annotations_url is None:
            print(f"No annotations for {external_id}")
            continue

        mp4_save_path = save_root / "videos" / external_id.with_suffix(".mp4")
        csv_save_path = save_root / "annotations" / external_id.with_suffix(".csv")
        viz_save_path = save_root / "viz_results" / external_id.with_suffix(".mp4")

        mp4_save_path.parent.mkdir(parents=True, exist_ok=True)
        csv_save_path.parent.mkdir(parents=True, exist_ok=True)
        viz_save_path.parent.mkdir(parents=True, exist_ok=True)

        download_video(video_url, mp4_save_path)
        bbdf = download_annotations(annotations_url, csv_save_path)
        bbdf.visualize_frames(mp4_save_path, viz_save_path)
