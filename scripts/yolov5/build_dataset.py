"""
Script to convert soccertrack dataset format to YOLOv5 format.
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from joblib import Parallel, delayed

import soccertrack
from soccertrack import Camera
from soccertrack.logger import logger, tqdm


# Move each file to their respective folderv and rename them to include their parent folder name
def move_files_to_folder(files, folder):
    folder.mkdir(exist_ok=True, parents=True)
    for file in files:
        file.rename(folder / (file.parent.stem + file.name))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset")
    parser.add_argument(
        "--save_dir", type=str, help="Path to save the dataset, must be a directory"
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    save_dir = Path(args.save_dir)

    assert len(list(save_dir.glob("*"))) == 0, "Save directory must be empty"

    paths_to_csv = sorted(dataset_path.glob("annotations/*.csv"))
    paths_to_mp4 = sorted(dataset_path.glob("videos/*.mp4"))

    for path_to_csv, path_to_mp4 in tqdm(
        zip(paths_to_csv, paths_to_mp4), total=len(paths_to_csv)
    ):
        # Create a camera object to read the width and height of the video
        cam = Camera(path_to_mp4)

        bbdf = soccertrack.load_df(path_to_csv)
        yolov5_formatted_data = bbdf.to_yolov5_format(
            save_dir=f"{save_dir}/{path_to_csv.stem}", w=cam.w, h=cam.h
        )

        logger.info(
            "Shape of yolov5_formatted_data:", np.array(yolov5_formatted_data).shape
        )
        logger.info("Sample of yolov5_formatted_data:", yolov5_formatted_data[0][:5])
        logger.info(
            f"Contents of {save_dir}:", *[p.name for p in save_dir.iterdir()][:5], "..."
        )

        def _parallel_imwrite(frame_num, frame):
            file_path = f"{save_dir}/{path_to_csv.stem}/{frame_num:06d}.png"
            cv2.imwrite(file_path, frame)

        res = Parallel(n_jobs=-1)(
            delayed(_parallel_imwrite)(frame_num, frame)
            for frame_num, frame in enumerate(tqdm(cam.iter_frames()))
        )
        break

    # Partition the dataset
    sequences = sorted(save_dir.glob("*"))

    seqs_dict = {
        "train": sequences[: int(len(sequences) * 0.7)],
        "val": sequences[int(len(sequences) * 0.7) : int(len(sequences) * 0.85)],
        "test": sequences[int(len(sequences) * 0.85) :],
    }

    for dataset, seqs in seqs_dict.items():
        for seq in seqs:
            move_files_to_folder(seq.glob("*.png"), save_dir / "images" / dataset)
            move_files_to_folder(seq.glob("*.txt"), save_dir / "labels" / dataset)
            seq.rmdir()

    dict_file = {
        "train": str((save_dir / "images/train").absolute()),
        "val": str((save_dir / "images/val").absolute()),
        "test": str((save_dir / "images/test").absolute()),
        "nc": 2,
        "names": ["player", "ball"],
    }

    yaml_path = (save_dir / "soccertrack_data.yaml").absolute()
    with open(yaml_path, "w") as file:
        documents = yaml.dump(dict_file, file)
