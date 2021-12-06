import os
from datetime import datetime
from typing import Mapping

import cv2 as cv
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm.auto import tqdm
from vidgear.gears import WriteGear

OmegaConf.register_new_resolver(
    "now", lambda x: datetime.now().strftime(x), replace=True
)


def load_config(yaml_path):
    assert os.path.exists(yaml_path)
    cfg = OmegaConf.load(yaml_path)

    cfg.outdir = cfg.outdir  # prevent multiple interpolations
    os.makedirs(cfg.outdir, exist_ok=True)

    # TODO: add valdation
    return cfg


def write_config(yaml_path: str, cfg: Mapping):
    assert os.path.exists(yaml_path)
    OmegaConf.save(cfg, yaml_path)


def pil2cv(image):
    """PIL型 -> OpenCV型"""
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv.cvtColor(new_image, cv.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv.cvtColor(new_image, cv.COLOR_RGBA2BGRA)
    return new_image


def cv2pil(image):
    """OpenCV型 -> PIL型"""
    new_image = image.copy()
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv.cvtColor(new_image, cv.COLOR_BGR2RGB)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv.cvtColor(new_image, cv.COLOR_BGRA2RGBA)
    new_image = Image.fromarray(new_image)
    return new_image


# Cell


def make_video(frames, fps, outpath):
    writer = WriteGear(output_filename=outpath, compression_mode=True)

    # loop over
    for frame in tqdm(frames):

        # simulating RGB frame for example
        frame_rgb = frame[:, :, ::-1]

        # writing RGB frame to writer
        writer.write(frame_rgb, rgb_mode=True)  # activate RGB Mode

    writer.close()


# Cell
class MovieIterator:
    """Very simple iterator class for movie files."""

    def __init__(self, path):
        vcInput = cv.VideoCapture(path)
        self.vcInput = vcInput
        self.video_fps = int(vcInput.get(cv.CAP_PROP_FPS))
        self.video_frame_count = int(vcInput.get(cv.CAP_PROP_FRAME_COUNT))
        self.img_width = int(vcInput.get(cv.CAP_PROP_FRAME_WIDTH))
        self.img_height = int(vcInput.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.path = path
        self._index = 0

    def __len__(self):
        return self.video_frame_count

    def __iter__(self):
        return self

    def __next__(self):
        if self._index < len(self):
            ret, img = self.vcInput.read()
            if ret:
                self._index += 1
                return img
            else:
                print("Unexpected end.")
                raise StopIteration
        raise StopIteration


def make_video(frames, fps, outpath):
    writer = WriteGear(output_filename=outpath, compression_mode=True)

    # loop over
    for frame in tqdm(frames):

        # simulating RGB frame for example
        frame_rgb = frame[:, :, ::-1]

        # writing RGB frame to writer
        writer.write(frame_rgb, rgb_mode=True)  # activate RGB Mode

    writer.close()


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
