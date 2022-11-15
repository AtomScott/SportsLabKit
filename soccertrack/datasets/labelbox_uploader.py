from __future__ import annotations

import uuid
from typing import Dict, Any

from labelbox import Client, LabelingFrontend, Dataset, Project
from labelbox.schema.ontology import OntologyBuilder, Tool
from labelbox.schema.media_type import MediaType

from pathlib import Path
import soccertrack
from soccertrack.logging import show_df # This just makes the df viewable in the notebook.
from tqdm import tqdm

def bbdf2ndjson(bbdf_file_name: str, data_row, schema_lookup:dict, keyframe_window: int) -> dict:
    """Converts a bounding box dataframe to ndjson format for labelbox import.
    
    Args:
        bbdf_file_name(str): The name of the bounding box dataframe file.
        data_row(): a clip information
        shema_lookup(dict): a dictionary that maps the class names to the labelbox schema
        keyframe_window(int): Interval between keyframes to be included in the video    
        
    Returns:
        ndjson_dict(dict) : A dictionary with the ndjson format for labelbox import.
    
    
    """
    
    bbdf = soccertrack.load_df(bbdf_file_name)
    bbdf.index = [i+1 for i in range(len(bbdf))]
    for (team_id, player_id), player_df in bbdf.iter_players():
        
        if team_id == '3':
            feature_name = 'BALL'
        elif team_id == '1' and int(player_id) >= 11:
            feature_name = team_id + '_' + str(int(player_id) - 11)
        elif team_id == '0' and player_id == '21':
            feature_name = '1' + '_' + str(int(player_id) - 11)
        else:
            feature_name = team_id + '_' + str(int(player_id))
            
        segments = []
        key_frames_dict = dict()
        key_frames_dict["keyframes"] = []
                
        for idx, row in player_df.iterrows():
            if idx % keyframe_window == 0:
                key_frames_dict["keyframes"].append({
                    "frame": idx,
                    "bbox": {
                        "top": int(row['bb_top']),
                        "left": int(row['bb_left']),
                        "height": int(row['bb_height']),
                        "width": int(row['bb_width'])
                    }
                })
            if idx == len(player_df)-1:
                key_frames_dict["keyframes"].append({
                    "frame": idx,
                    "bbox": {
                        "top": int(row['bb_top']),
                        "left": int(row['bb_left']),
                        "height": int(row['bb_height']),
                        "width": int(row['bb_width'])
                    }
                })  

        segments.append(key_frames_dict)
        
        ndjson_dict =  {
        "uuid": str(uuid.uuid4()),
        "schemaId": schema_lookup[feature_name],
        "dataRow": {
            "id": data_row.uid
        },
        "segments": segments
        }
        return ndjson_dict

def main():
    # Add your api key
    API_KEY='Enter your API key here'
    client = Client(api_key=API_KEY)


    PROJECT_NAME = "Enter your project name here" # This is the name of the project you want to upload the data to.
    DATASET_NAME = 'Enter your dataset name here' # This is the name of the dataset.
    #get project information
    project = next(client.get_projects(where=Project.name == PROJECT_NAME), None)
    #get dataset information
    dataset = next(client.get_datasets(where=Dataset.name == DATASET_NAME), None)
    
    # We want to try out a few different tools here.
    ontology_builder = OntologyBuilder(
    tools=[Tool(tool=Tool.Type.BBOX, name="Enter your label tool name here")]) # This is the name of the label tool you want to use.
    
    # When we created a project with the ontology defined above, all of the ids were assigned.
    # So lets reconstruct the ontology builder with all of the ids.
    ontology = ontology_builder.from_project(project)
    # We want all of the feature schemas to be easily accessible by name.
    schema_lookup = {tool.name: tool.feature_schema_id for tool in ontology.tools}
    
    keyframe_window = 1
    input_csv_file = 'labelbox_data/annotations/'
    for data_row in dataset.data_rows():
        file_name = Path(input_csv_file) / f"{data_row.external_id.split('.')[0]}.csv"
        uploads = []
        try:
            uploads.append(bbdf2ndjson(file_name, data_row, schema_lookup, keyframe_window))
        except FileNotFoundError: # If the file doesn't exist, we'll skip it.
            print('FileNotFoundError', data_row.external_id)
            continue

        # Let's upload!
        # Validate must be set to false for video bounding boxes
        upload_task = project.upload_annotations(name=f"upload-job-{uuid.uuid4()}",
                                                annotations=uploads,
                                                validate=False)

        # Wait for upload to finish (Will take up to five minutes)
        upload_task.wait_until_done()
        # Review the upload status
        print('Done!', data_row.external_id, ': Errors', upload_task.errors)

if __name__ == '__main__':
    main()