"""Genereal utils."""
import os
from datetime import datetime
from typing import Iterable
import cv2 as cv
import numpy as np
from numpy.typing import NDArray
from omegaconf import OmegaConf
from PIL import Image
from vidgear.gears import WriteGear

from soccertrack.utils import logger, tqdm

OmegaConf.register_new_resolver(
    "now", lambda x: datetime.now().strftime(x), replace=True
)


def load_config(yaml_path: str) -> OmegaConf:
    """Load config from yaml file.

    Args:
        yaml_path (str): Path to yaml file

    Returns:
        OmegaConf: Config object loaded from yaml file
    """
    assert os.path.exists(yaml_path)
    cfg = OmegaConf.load(yaml_path)

    cfg.outdir = cfg.outdir  # prevent multiple interpolations
    os.makedirs(cfg.outdir, exist_ok=True)

    # TODO: add validation
    return cfg


def write_config(yaml_path: str, cfg: OmegaConf) -> None:
    """Write config to yaml file.

    Args:
        yaml_path (str): Path to yaml file
        cfg (OmegaConf): Config object
    """
    assert os.path.exists(yaml_path)
    OmegaConf.save(cfg, yaml_path)


def pil2cv(image: Image.Image) -> NDArray[np.uint8]:
    """Convert PIL image to OpenCV image.

    Args:
        image (Image.Image): PIL image

    Returns:
        NDArray[np.uint8]: Numpy Array (OpenCV image)
    """
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv.cvtColor(new_image, cv.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv.cvtColor(new_image, cv.COLOR_RGBA2BGRA)
    return new_image


def cv2pil(image: NDArray[np.uint8]) -> Image.Image:
    """Convert OpenCV image to PIL image.

    Args:
        image (NDArray[np.uint8]): Numpy Array (OpenCV image)

    Returns:
        Image.Image: PIL image
    """
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv.cvtColor(new_image, cv.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv.cvtColor(new_image, cv.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


def make_video(frames: Iterable[NDArray[np.uint8]], outpath: str) -> None:
    """Make video from a list of opencv format frames.

    Args:
        frames (Iterable[NDArray[np.uint8]]): List of opencv format frames
        outpath (str): Path to output video file

    Todo:
        * add FPS option
        * functionality to use PIL image
        * reconsider compression (current compression is not good)
    """
    writer = WriteGear(output_filename=outpath, compression_mode=True)

    # loop over
    for frame in tqdm(frames):

        # simulating RGB frame for example
        frame_rgb = frame[:, :, ::-1]

        # writing RGB frame to writer
        writer.write(frame_rgb, rgb_mode=True)  # activate RGB Mode

    writer.close()


class MovieIterator:
    def __init__(self, path: str):
        """Very simple iterator class for movie files.

        Args:
            path (str): Path to movie file

        Attributes:
            video_fps (int): Frames per second
            video_frame_count (int): Total number of frames
            vcInput (cv.VideoCapture): OpenCV VideoCapture object
            img_width (int): Width of frame
            img_height (int): Height of frame
            path (str): Path to movie file

        Raises:
            FileNotFoundError: If file does not exist

        """
        if not os.path.isfile(path):
            raise FileNotFoundError

        vcInput = cv.VideoCapture(path)
        self.vcInput = vcInput
        self.video_fps: int = int(vcInput.get(cv.CAP_PROP_FPS))
        self.video_frame_count = int(vcInput.get(cv.CAP_PROP_FRAME_COUNT))
        self.img_width = int(vcInput.get(cv.CAP_PROP_FRAME_WIDTH))
        self.img_height = int(vcInput.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.path = path
        self._index = 0

    def __len__(self) -> int:
        return self.video_frame_count

    def __iter__(self) -> "MovieIterator":
        return self

    def __next__(self) -> NDArray[np.uint8]:
        if self._index < len(self):
            ret, img = self.vcInput.read()
            if ret:
                self._index += 1
                return img
            logger.debug("Unexpected end.")  # <- Not sure why this happens
        raise StopIteration


class ImageIterator:
    def __init__(self, path: str):
        """Very simple iterator class for image files.

        Args:
            path (str): Path to image file
        """
        assert os.path.isdir(path), f"{path} is not a directory."
        self.path = path

        imgs = []
        valid_images = [".jpg", ".gif", ".png", ".tga"]
        for f in os.listdir(path):
            ext = os.path.splitext(f)[1]
            if ext.lower() not in valid_images:
                continue
            imgs.append(cv.imread(os.path.join(path, f)))
        self.imgs = imgs
        self._index = 0

    def __len__(self) -> int:
        return len(self.imgs)

    def __iter__(self) -> "ImageIterator":
        return self

    def __next__(self) -> NDArray[np.uint8]:
        if self._index < len(self):
            img = self.imgs[self._index]
            self._index += 1
            return img
        raise StopIteration


# Due to memory consumption concerns, the function below has been replaced by the function that uses vidgear above.
# ===
# def make_video(images: list, fps: int, outpath: str = 'video.mp4'):
#     """The main def for creating a temporary video out of the
#     PIL Image list passed, according to the FPS passed
#     Parameters
#     ----------
#     image_list : list
#         A list of PIL Images in sequential order you want the video to be generated
#     fps : int
#         The FPS of the video
#     """

#     def convert(img):
#         if isinstance(img, Image.Image):
#             return pil2cv(img)
#         elif isinstance(img, np.ndarray):
#             return img
#         else:
#             raise ValueError(type(img))

#     h, w = convert(images[0]).shape[:2]
#     fourcc = cv.VideoWriter_fourcc('M','J','P','G')
#     video = cv.VideoWriter(filename=outpath+'.mp4', fourcc=fourcc, fps=fps, frameSize=(w, h))

#     for img in tqdm(images, total=len(images)):
#         video.write(img)
#     video.release()
#     os.system(f"ffmpeg -i {outpath+'.mp4'} -vcodec libx264 -acodec aac -y {outpath}")
#     print(f"Find your images and video at {outpath}")
