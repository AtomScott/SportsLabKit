from glob import glob
import os
import gdown
from click import command, group, option

from soccertrack.utils.camera import find_intrinsic_camera_parameters, Camera
from soccertrack.utils.utils import make_video
from soccertrack.utils import logger, set_log_level


@group()
def cli():
    """ """
    pass


@cli.command()
@option("--log-level", default="INFO")
def test_logger(log_level):
    set_log_level(log_level)
    logger.debug("That's it, beautiful and simple logging!")
    logger.info("This is an info message")
    logger.success("success!")
    logger.warning("I am warning you Github copilot!")
    logger.error("I am error you Github copilot!")
    logger.critical("Fire in the hole!")


@cli.command()
@option("-d", "--dataset", default="all", help="Which data to download.")
@option("-o", "--output", default="./data", help="Where to save the data.")
def download(dataset, output, quiet=False):
    if dataset == "all":
        url = "https://drive.google.com/drive/u/1/folders/13bk0oSsH0WL9LBmr9_4zYn6WqfntT3qF"
    else:
        raise ValueError("Dataset not found.")

    gdown.download_folder(url=url, output=output, quiet=quiet, use_cookies=False)


@cli.command()
@option(
    "-i",
    "--input",
    required=True,
    help="Path to the input video (wildcards are supported).",
)
@option(
    "-c",
    "--checkerboard",
    required=True,
    help="Path to the checkerboard video (wildcards are supported).",
)
@option("-o", "--output", required=True, help="Path to the output video.")
@option(
    "-f",
    "--fps",
    default=30,
    help="Number of frames per second to use for calibration.",
)
@option(
    "-s",
    "--scale",
    default=1,
    help="Scale factor for the checkerboard. Scales the checkerboard video by 1/s.",
)
@option("-p", "--pts", default=50, help="Number of points to use for calibration.")
@option("--calibration_method", default="zhang", help="Calibration method.")
@option("--keypoint_xml", default=None, help="Path to the keypoint xml file.")
def calibrate(
    input, checkerboard, output, fps=1, scale=1, pts=50, calibration_method="zhang", keypoint_xml=None
):
    """Calibrate video from input"""
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


if __name__ == "__main__":
    cli()
