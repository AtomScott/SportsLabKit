from __future__ import annotations

import argparse
import os
import uuid
from pathlib import Path
from time import time

from dotenv import load_dotenv
from labelbox import Client, Dataset, MALPredictionImport, Project
from labelbox.schema.ontology import OntologyBuilder, Tool

import soccertrack
from soccertrack.logger import tqdm

load_dotenv()

LABELBOX_API_KEY = os.getenv("LABELBOX_API_KEY")
KEYFRAME_WINDOW = 1  # Keyframe Interval to upload

client = Client(api_key=LABELBOX_API_KEY)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p", "--LABELBOX_PROJECT_NAME", help="Enter your project name here : "
)
parser.add_argument(
    "-d", "--LABELBOX_DATASET_NAME", help="Enter your dataset name here : "
)
parser.add_argument(
    "-i",
    "--INPUT_BBDF_DIR",
    help="Enter the name of the csv file that contains the bbdf information :  ",
)
args = parser.parse_args()


def upload_annotations(
    project: object, data_row: object, labelbox_data: object
) -> None:
    """Upload annotations to Labelbox as model assisted labels.

    Args:
        project (object): The Labelbox project to upload to.
        data_row (object): The Labelbox data row(single video) to upload to.
        labelbox_data (object): The Labelbox format data to upload.

    Returns:
        None
    """
    # timestampを入れたほうが親切かな
    # 同じ名前のtask_nameがあるとエラーになるので、一意な名前をつける意味も
    upload_task_name = f"upload-job-{data_row.external_id}-{time()}"

    # Use MAL since LabelImport has strict API rate limits
    upload_job = MALPredictionImport.create_from_objects(
        client=client,
        project_id=project.uid,
        name="mal_job" + str(uuid.uuid4()),
        predictions=labelbox_data,
    )

    # Wait for upload to finish
    upload_job.wait_until_done()

    # Review the upload status
    print(
        "Done!",
        " job name : ",
        upload_task_name,
        ": Errors",
        upload_job.errors,
    )


if __name__ == "__main__":
    # set up project information

    project = next(
        client.get_projects(where=Project.name == args.LABELBOX_PROJECT_NAME), None
    )
    dataset = next(
        client.get_datasets(where=Dataset.name == args.LABELBOX_DATASET_NAME), None
    )
    ontology = OntologyBuilder.from_project(project)
    schema_lookup = {tool.name: tool.feature_schema_id for tool in ontology.tools}

    # ソートすると0000からスタートする
    data_rows = sorted(dataset.data_rows(), key=lambda x: x.external_id)

    # create a sample upload
    data_rows = list(dataset.data_rows())[::-1]
    for data_row in tqdm(data_rows):
        file_name = f"{data_row.external_id.split('.')[0]}.csv"
        bbdf_file_path = Path(args.INPUT_BBDF_DIR) / file_name

        # load the bbdf and convert to labelbox format
        try:
            bbdf = soccertrack.load_df(bbdf_file_path)
            labelbox_data = bbdf.to_labelbox_data(
                data_row, schema_lookup, KEYFRAME_WINDOW
            )
            upload_annotations(project, data_row, labelbox_data)

        # If the file doesn't exist, we'll skip it.
        except FileNotFoundError:
            print("FileNotFoundError", data_row.external_id)
            continue
