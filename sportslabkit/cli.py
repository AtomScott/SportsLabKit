import os
from glob import glob
from typing import Optional

import numpy as np
from fire import Fire

from sportslabkit.utils import logger, set_log_level
from sportslabkit.utils.camera import Camera, find_intrinsic_camera_parameters


class CLI:
    """CLI for soccertrack."""

    def test_logger(self, log_level: str = "INFO"):
        """Test the output of soccertrack's logger

        Args:
            log_level (str): The log level to use. Defaults to "INFO".
        """
        set_log_level(log_level)
        logger.debug("That's it, beautiful and simple logging!")
        logger.info("This is an info message")
        logger.success("success!")
        logger.warning("I am warning you Github copilot!")
        logger.error("I am error you Github copilot!")
        logger.critical("Fire in the hole!")

    def download(
        self, dataset: str = "all", output: str = "./data", quiet: bool = False
    ):
        """Download data from google drive

        Args:
            dataset (str, optional): Dataset to download. Defaults to "all".
            output (str, optional): Where to save the data. Defaults to "./data".
            quiet (bool, optional): Whether to silence the output. Defaults to False.

        Raises:
            ValueError: _description_
        """
        if dataset == "all":
            url = "https://drive.google.com/drive/u/1/folders/13bk0oSsH0WL9LBmr9_4zYn6WqfntT3qF"
        else:
            raise ValueError("Dataset not found.")

        gdown.download_folder(url=url, output=output, quiet=quiet, use_cookies=False)

    def find_calibration_parameters(
        self,
        checkerboard_files: str,
        output: str,
        fps: int = 1,
        scale: int = 1,
        pts: int = 50,
        calibration_method: str = "zhang",
    ):
        """_summary_

        Args:
            checkerboard_files (str): Path to the checkerboard video (wildcards are supported).
            output (str): Path to save the calibration parameters.
            fps (int, optional): _description_. Defaults to 1.
            scale (int, optional): _description_. Defaults to 1.
            pts (int, optional): _description_. Defaults to 50.
            calibration_method (str, optional): _description_. Defaults to "zhang".
        """

        mtx, dist, mapx, mapy = find_intrinsic_camera_parameters(
            checkerboard_files,
            fps=fps,
            scale=scale,
            save_path=False,
            draw_on_save=False,
            points_to_use=pts,
            calibration_method=calibration_method,
            return_mappings=True,
        )
        dirname = os.path.dirname(output)
        if len(dirname) != 0:
            os.makedirs(dirname, exist_ok=True)
        # save the calibration parameters to output
        np.savez(output, mtx=mtx, dist=dist, mapx=mapx, mapy=mapy)

    def calibrate_from_npz(
        self,
        input: str,
        npzfile: str,
        output: str,
        calibration_method: str = "zhang",
        keypoint_xml: Optional[str] = None,
        **kwargs,
    ):
        """Calibrate a video using precomputed calibration parameters

        Args:
            input (str): _description_
            npzfile (str): _description_
            output (str): _description_
            calibration_method (str, optional): _description_. Defaults to "zhang".
            keypoint_xml (Optional[str], optional): _description_. Defaults to None.

        Note:
            kwargs are passed to `make_video`, so it is recommended that you refere to the documentation for `make_video`.

        """
        mtx, dist, mapx, mapy = np.load(npzfile).values()

        camera = Camera(
            video_path=input,
            keypoint_xml=keypoint_xml,
            x_range=None,
            y_range=None,
            calibration_method=calibration_method,
            camera_matrix=mtx,
            distortion_coefficients=dist,
        )
        camera.mapx = mapx
        camera.mapy = mapy

        if keypoint_xml is not None:
            camera.source_keypoints = camera.undistort_points(camera.source_keypoints)

        dirname = os.path.dirname(output)
        if len(dirname) != 0:
            os.makedirs(dirname, exist_ok=True)
        camera.save_calibrated_video(
            save_path=output,
            **kwargs,
        )
        logger.info(f"Video saved to {output}")

    def calibrate(
        self,
        input: str,
        checkerboard: str,
        output: str,
        fps: int = 1,
        scale: int = 1,
        pts: int = 50,
        calibration_method: str = "zhang",
        keypoint_xml: Optional[str] = None,
    ):
        """Calibrate video from input

        Args:
            input (str): Path to the input video (wildcards are supported).
            checkerboard (str): Path to the checkerboard video (wildcards are supported).
            output (str): Path to the output video.
            fps (int, optional): Number of frames per second to use for calibration. Defaults to 1.
            scale (int, optional): Scale factor for the checkerboard. Scales the checkerboard video by 1/s. Defaults to 1.
            pts (int, optional): Number of points to use for calibration. Defaults to 50.
            calibration_method (str, optional): Calibration method. Defaults to "zhang".
            keypoint_xml (Optional[str], optional): Path to the keypoint xml file. Defaults to None.
        """
        input_files = list(glob(input))
        checkerboard_files = list(glob(checkerboard))

        mtx, dist, mapx, mapy = find_intrinsic_camera_parameters(
            checkerboard_files,
            fps=fps,
            scale=scale,
            save_path=False,
            draw_on_save=False,
            points_to_use=pts,
            calibration_method=calibration_method,
            return_mappings=True,
        )

        for input_file in input_files:
            camera = Camera(
                video_path=input_file,
                keypoint_xml=keypoint_xml,
                x_range=None,
                y_range=None,
                calibration_method=calibration_method,
                calibration_video_path=checkerboard_files,
                camera_matrix=mtx,
                distortion_coefficients=dist,
            )
            camera.mapx = mapx
            camera.mapy = mapy
            camera.source_keypoints = camera.undistort_points(camera.source_keypoints)

            save_path = os.path.join(output, os.path.basename(input_file))
            camera.save_calibrated_video(save_path=save_path)
            logger.info(f"Video saved to {save_path}")


def main():
    cli = CLI()
    Fire(cli)


if __name__ == "__main__":
    cli = CLI()
    Fire(cli)
