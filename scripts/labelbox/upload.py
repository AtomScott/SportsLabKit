from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from labelbox import Client, Dataset, LabelingFrontend, Project
from labelbox.schema.media_type import MediaType
from labelbox.schema.ontology import OntologyBuilder, Tool
from tqdm import tqdm

import soccertrack
from soccertrack.logging import (  # This just makes the df viewable in the notebook.
    show_df,
)


def create_ndjson(
    datarow_id: str, schema_id: str, segments: Dict[str, Any]
) -> Dict[str, Any]:
    return {
        "uuid": str(uuid.uuid4()),
        "schemaId": schema_id,
        "dataRow": {"id": datarow_id},
        "segments": segments,
    }


def _fix_frame(
    bbdf_tmp,
    data_row_list,
    data_row,
    input_csv_file,
    SECOND_START_IDX,
    THRESHOLD_SECOND_START,
    THRESHOLD_SECOND_END,
    DATA_ROW_IDX,
):
    # データフレームのズレの修正
    start_second = int(data_row.external_id.split("_")[SECOND_START_IDX])
    if (
        start_second
        >= THRESHOLD_SECOND_START & start_second
        < THRESHOLD_SECOND_END - 30
    ):
        _data_row = list(reversed(data_row_list))[DATA_ROW_IDX + 1]
        _bbdf_file_name = (
            Path(input_csv_file) / f"{_data_row.external_id.split('.')[0]}.csv"
        )
        _bbdf = soccertrack.load_df(_bbdf_file_name)
        bbdf = pd.concat([bbdf_tmp[2:], _bbdf[0:2]], axis=0)

    elif start_second >= THRESHOLD_SECOND_END - 30:
        bbdf = bbdf_tmp[2:]

    else:
        bbdf = bbdf_tmp
    bbdf.index = [i + 1 for i in range(len(bbdf))]
    return bbdf


def get_segment(bbdf, KEYFRAME_WINDOW):
    segment = dict()
    for (team_id, player_id), player_df in bbdf.iter_players():

        if team_id == "3":
            feature_name = "BALL"
        elif team_id == "1" and int(player_id) >= 11:
            feature_name = team_id + "_" + str(int(player_id) - 11)
        elif team_id == "0" and player_id == "21":
            feature_name = "1" + "_" + str(int(player_id) - 11)
        elif team_id == "0" and player_id == "11":
            feature_name = "0" + "_" + str(int(player_id) - 11)

        else:
            feature_name = team_id + "_" + str(int(player_id))

        key_frames_dict = dict()
        key_frames_dict["keyframes"] = []

        for idx, row in player_df.iterrows():
            if idx % KEYFRAME_WINDOW == 0:
                try:
                    key_frames_dict["keyframes"].append(
                        {
                            "frame": idx,
                            "bbox": {
                                "top": int(row["bb_top"]),
                                "left": int(row["bb_left"]),
                                "height": int(row["bb_height"]),
                                "width": int(row["bb_width"]),
                            },
                        }
                    )
                except ValueError:
                    print("ValueError occured :", feature_name, "frame_num :", idx)

        segment[feature_name] = [key_frames_dict]
    return segment


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--API_key", help="Enter your API key here : ")
    parser.add_argument("--PROJECT_NAME", help="Enter your project name here : ")
    parser.add_argument("--DATASET_NAME", help="Enter your dataset name here : ")
    parser.add_argument("--ONTOLOGY_NAME", help="Enter your ontology name here : ")
    parser.add_argument(
        "--input_csv_file",
        help="Enter the name of the csv file that contains the bbdf information :  ",
    )

    args = parser.parse_args()

    API_KEY = args.API_key  # Add your api key
    PROJECT_NAME = (
        args.PROJECT_NAME
    )  # This is the name of the project you want to upload the data to.
    DATASET_NAME = args.DATASET_NAME  # This is the name of the dataset.
    ONTOLOGY_NAME = args.ONTOLOGY_NAME  # This is the name of the ontology.

    client = Client(api_key=API_KEY)
    # get project information
    project = next(client.get_projects(where=Project.name == PROJECT_NAME), None)
    # get dataset information
    dataset = next(client.get_datasets(where=Dataset.name == DATASET_NAME), None)
    # We want to try out a few different tools here.
    ontology_builder = OntologyBuilder(
        tools=[Tool(tool=Tool.Type.BBOX, name=ONTOLOGY_NAME)]
    )  # This is the name of the label tool you want to use.

    # When we created a project with the ontology defined above, all of the ids were assigned.
    # So lets reconstruct the ontology builder with all of the ids.
    ontology = ontology_builder.from_project(project)
    # We want all of the feature schemas to be easily accessible by name.
    schema_lookup = {tool.name: tool.feature_schema_id for tool in ontology.tools}

    KEYFRAME_WINDOW = 1
    SECOND_START_IDX = 3
    THRESHOLD_SECOND_START = 840
    THRESHOLD_SECOND_END = 1800

    data_row_list = [data_row for data_row in dataset.data_rows()]
    input_csv_file = (
        args.input_csv_file
    )  # This is the name of the csv file that contains the video information.

    for DATA_ROW_IDX in tqdm(range(len(data_row_list))):

        data_row = list(reversed(data_row_list))[DATA_ROW_IDX]
        bbdf_file_name = (
            Path(input_csv_file) / f"{data_row.external_id.split('.')[0]}.csv"
        )
        try:
            bbdf_tmp = soccertrack.load_df(bbdf_file_name)
        except FileNotFoundError:  # If the file doesn't exist, we'll skip it.
            print("FileNotFoundError", data_row.external_id)
            continue

        # Correction of data frame misalignment
        bbdf = _fix_frame(
            bbdf_tmp,
            data_row_list,
            data_row,
            input_csv_file,
            SECOND_START_IDX,
            THRESHOLD_SECOND_START,
            THRESHOLD_SECOND_END,
            DATA_ROW_IDX,
        )
        segment = get_segment(bbdf, KEYFRAME_WINDOW)

        uploads = []
        for schema_name, schema_id in schema_lookup.items():
            if schema_name in segment:
                uploads.append(
                    create_ndjson(data_row.uid, schema_id, segment[schema_name])
                )
        upload_task = project.upload_annotations(
            name=f"upload-job-{uuid.uuid4()}", annotations=uploads, validate=False
        )
        # Wait for upload to finish (Will take up to five minutes)
        upload_task.wait_until_done()
        # Review the upload status
        print(
            "Done!",
            " Video_name : ",
            data_row.external_id,
            ": Errors",
            upload_task.errors,
        )


if __name__ == "__main__":
    main()
