from glob import glob
import os
import gdown
from click import command, group, option

from soccertrack.utils.camera import find_intrinsic_camera_parameters, Camera
from soccertrack.utils.utils import make_video
from soccertrack.utils import logger, set_log_level

@group()
def cli():
    pass

@cli.command()
@option('--log-level', default='INFO')
def test_logger(log_level):
    set_log_level(log_level)
    logger.debug("That's it, beautiful and simple logging!")
    logger.info("This is an info message")
    logger.success("success!")
    logger.warning("I am warning you Github copilot!")
    logger.error("I am error you Github copilot!")
    logger.critical("Fire in the hole!")


@cli.command()
@option('-d', '--dataset', default='all', help='Which data to download.')
@option('-o', '--output', default='./data', help='Where to save the data.')
def download(dataset, output, quiet=False):
    if dataset == 'all':
        url = 'https://drive.google.com/drive/u/1/folders/13bk0oSsH0WL9LBmr9_4zYn6WqfntT3qF'
    else:  
        raise ValueError('Dataset not found.')

    gdown.download_folder(url=url, output=output, quiet=quiet, use_cookies=False)

        
@cli.command()
@option('-i', '--input', required=True, help='Path to the input video (wildcards are supported).')
@option('-c', '--checkerboard', required=True, help='Path to the checkerboard video (wildcards are supported).')
@option('-o', '--output', required=True, help='Path to the output video.')
def calibrate(input, checkerboard, output):
    input_files = list(glob(input))
    checkerboard_files = list(glob(checkerboard))
    
    camera_matrix, distortion_coefficients=find_intrinsic_camera_parameters(
        input_files, 
        fps=1, 
        s=10, 
        save_path=output, 
        draw_on_save=True
    )
    
    for input_file in input_files:
        camera = Camera(
            video_path=input_file,
            keypoint_xml=None,
            x_range=None,
            y_range=None,
            camera_matrix=camera_matrix,
            camera_matrix_path=None,
            distortion_coefficients=distortion_coefficients,
            distortion_coefficients_path=None,
        )
        save_path = os.path.join(output, os.path.basename(input_file))
        camera.save_calibrated_video(save_path=save_path)




if __name__ == '__main__':
    cli()