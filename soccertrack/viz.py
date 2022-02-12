import os
import io
import cv2 as cv
import subprocess
from itertools import chain
from tempfile import TemporaryDirectory
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
from mplsoccer import Pitch
from numpy.lib.npyio import save

from soccertrack.utils import logger, tqdm
from soccertrack.utils.camera import Camera
from soccertrack.utils.detection import CandidateDetection
from soccertrack.utils.utils import make_video


def visualize_cameras(
    cameras: Iterable[Camera],
    candidate_detections: Mapping[int, Iterable[CandidateDetection]],
    save_dir: str = "",
    auto_grid: bool = True,
    **kwargs,
):
    """Visualize cameras.

    Args:
        cameras (Iterable[Camera]): [description]
        candidate_detections (Mapping[int, Iterable[CandidateDetection]]): [description]
        save_dir (str, optional): [description]. Defaults to ''.
        auto_grid (bool, optional): Generate a grid of videos based on the shape of `cameras`. Defaults to True.
            For example, if `cameras` is a 1d-list of 4 cameras, the grid will be 1x4 (h w). If `cameras` is a 2d-list
            of 2 cameras each, the grid will be 2x2.

    Note:
        kwargs are passed to `visualize_candidate_detections` and `make_video`, so it
        is recommended that you refer to the documentation of each function.

    """

    # create save directory if it does not exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    logger.debug(f"Saving visualizations to {save_dir}")

    # Make video with temporary directory
    with TemporaryDirectory() as tmpdir:
        tmp_paths = []
        for i, camera in enumerate(cameras):

            tmp_path = os.path.join(tmpdir, f"{camera.label}-{i:02d}.mp4")
            tmp_paths.append(tmp_path)

            logger.info(f"Visualizing camera {camera.label}")
            camera.visualize_candidate_detections(
                candidate_detections=candidate_detections, save_path=tmp_path, **kwargs
            )

        if auto_grid:
            if kwargs.get("height", 1) + kwargs.get("width", 1) == 2:
                logger.warning(
                    "Auto-grid requires height and/or width to be the equal for all cameras."
                )

            n_rows, n_cols = (
                np.shape(cameras) if np.ndim(cameras) == 2 else (1, len(cameras))
            )
            logger.info(f"Auto-generating video of shape {n_rows}x{n_cols}")

            input_videos = chain.from_iterable(
                ["-i", tmp_path] for tmp_path in tmp_paths
            )

            if n_rows == 1:
                filter_complex = f"hstack=inputs={n_cols}"
            elif n_cols == 1:
                filter_complex = f"vstack=inputs={n_rows}"
            else:
                filter_complex = '"'
                for ri in range(n_rows):
                    filter_complex += "".join(
                        [f"[{ci + ri * n_cols}:v]" for ci in range(n_cols)]
                    )
                    filter_complex += f"hstack=inputs={n_cols}[row{ri}];"
                    filter_complex += "".join([f"[row{ri}]" for ri in range(n_rows)])
                    filter_complex += f"vstack=inputs={n_rows}[v];"
                filter_complex += '" -map "[v]"'

            save_path = os.path.join(save_dir, "grid.mp4")

            cmd = " ".join(
                [
                    "ffmpeg",
                    *input_videos,
                    "-filter_complex",
                    filter_complex,
                    "-y",  # overwrite
                    save_path,
                ]
            )

            logger.debug(f"ffmpeg command: \n{cmd}")
            output = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL)
            logger.debug(f"ffmpeg output: \n{output}")
            logger.info(f"Saving video to {save_path}")

        else:
            for tmp_path in tmp_paths:
                logger.info(f"Saving {os.path.basename(tmp_path)}")
                os.rename(tmp_path, os.path.join(save_dir, os.path.basename(tmp_path)))

def get_xsys(detections: Iterable[CandidateDetection], cameras:Iterable[Camera]):
    if isinstance(detections, dict):
        detections = list(detections.values())

    xs, ys = [], []
    for dets in detections:
        if isinstance(dets, CandidateDetection):
            dets = [dets]
        else:
            raise ValueError(f"{dets} is not a CandidateDetection or a list of CandidateDetections")
        for pd in dets:
            if pd.camera.label == cameras[0].label:
                px, py = cameras[0].video2pitch(np.array([pd._x, pd._y])).squeeze()
            elif pd.camera.label == cameras[1].label:
                px, py = cameras[1].video2pitch(np.array([pd._x, pd._y])).squeeze()
            else:
                raise ValueError("camera label not found")

            assert px == pd.px, (px, pd.px)
            assert py == pd.py, (py, pd.py)
            xs.append(px)
            ys.append(py)
    return xs, ys

def visualize_cameras_to_pitch(
    cameras: Iterable[Camera],
    candidate_detections: Mapping[int, Iterable[CandidateDetection]],
    save_path: str = "",
    plot_keypoints: bool = False,
    auto_grid: bool = True,
    afterimage: int = 5,
    **kwargs,
):

    pitch = Pitch(
        pitch_color="black",
        line_color=(0.3, 0.3, 0.3),
        pitch_type="custom",
        pitch_length=105,
        pitch_width=68,
    )

    if plot_keypoints:
        kxs,kys =[],[]
        for camera in cameras:
            for x, y in camera.source_keypoints:
                x, y = camera.video2pitch(np.array([x, y])).squeeze()
                kxs.append(x)
                kys.append(y)

    frames = []
    prev_xs, prev_ys = [], []
    for candidate_detections_per_frame in tqdm(candidate_detections.values(), level="DEBUG", desc="Drawing pitch"):
        xs, ys = get_xsys(candidate_detections_per_frame, cameras)

        fig, ax = pitch.draw()
        ax.scatter(xs, ys, color="deeppink")
        if plot_keypoints: ax.scatter(kxs, kys, color="red")
        
        len_afterimage = min(afterimage, len(prev_xs))
        if len_afterimage > 0:
            size = np.linspace(0.3, 1, len_afterimage) * 20
            alpha = np.linspace(0.2, 1, len_afterimage)

            for i in range(len_afterimage):
                ax.scatter(prev_xs[-i], prev_ys[-i], color="deeppink", s=size[-i], alpha=alpha[-i])
        fig.canvas.draw()
        
        with io.BytesIO() as buff:
            fig.savefig(buff, format='raw')
            buff.seek(0)
            data = np.frombuffer(buff.getvalue(), dtype=np.uint8)
        w, h = fig.canvas.get_width_height()
        img = data.reshape((int(h), int(w), -1))[:, :, ::-1]

        frames.append(img)
        plt.close()
        prev_xs.append(xs)
        prev_ys.append(ys)


    make_video(frames, outpath=save_path)

