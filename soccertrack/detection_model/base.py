from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import requests
from PIL import Image

from soccertrack.types import Detection
from soccertrack.utils.draw import draw_bounding_boxes


def read_image(img):
    """Reads an image from a file, URL, or a numpy array.
    Args:
        img (str, Path, Image.Image, or np.ndarray): The image to read.
    Returns:
        np.ndarray: The image as a numpy array.
    """
    if isinstance(img, str):
        if img.startswith("http"):
            img = requests.get(img, stream=True).raw
            img = Image.open(img)
        else:
            img = Path(img)
    if isinstance(img, Path):
        img = Image.open(img)
    if isinstance(img, Image.Image):
        img = np.array(img)
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Unsupported input type: {type(img)}")
    if len(img.shape) != 3:
        raise ValueError(f"Unsupported input shape: {img.shape}")
    if img.shape[2] not in [1, 3]:
        raise ValueError(f"Unsupported input shape: {img.shape}")

    return img


# class Detection:
#     def __init__(
#         self,
#         box: np.ndarray,
#         score: Optional[float] = None,
#         class_id: Optional[int] = None,
#         feature: Optional[np.ndarray] = None,
#     ):
#         self.box = box
#         self.score = score
#         self.class_id = class_id
#         self.feature = feature

#     def __repr__(self):
#         return f"Detection(box={self.box}, score={self.score:.5f}, class_id={self.class_id}, feature={self.feature})"


class Detections:
    def __init__(
        self, pred, im, file=None, cam=None, times=(0, 0, 0), names=None, shape=None
    ):
        """SoccerTrack detections class for inference results."""

        # normalizations
        self.im = self._process_im(im)  # image as numpy array
        self.pred = self._process_pred(pred)  # tensors pred[0] = (xywh, conf, cls)

        self.names = names  # class names
        self.times = times  # profiling times

    def _process_im(self, im):
        return read_image(im)

    def _process_pred(self, pred):
        # process predictions
        if isinstance(pred, dict):
            if len(pred.keys()) == 0:
                return np.array([])
            pred = np.stack(
                [
                    pred["bbox_left"],
                    pred["bbox_top"],
                    pred["bbox_width"],
                    pred["bbox_height"],
                    pred["conf"],
                    pred["class"],
                ],
                axis=1,
            )
        return pred

    def show(self, **kwargs):
        im = self.im
        boxes = self.pred[:, :4]
        labels = [f"{int(c)} {conf:.2f}" for conf, c in self.pred[:, 4:]]
        draw_im = draw_bounding_boxes(im, boxes, labels, **kwargs)
        return Image.fromarray(draw_im)

    def save(self, labels=True, save_dir="runs/detect/exp", exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True)  # increment save_dir
        self._run(save=True, labels=labels, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir="runs/detect/exp", exist_ok=False):
        save_dir = increment_path(save_dir, exist_ok, mkdir=True) if save else None
        return self._run(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self, labels=True):
        self._run(render=True, labels=labels)  # render results
        return self.ims

    def to_df(self):
        # return detections as pandas DataFrames, i.e. print(results.to_df())
        df = pd.DataFrame(
            self.pred,
            columns=[
                "bbox_left",
                "bbox_top",
                "bbox_width",
                "bbox_height",
                "conf",
                "class",
            ],
        )
        return df

    def to_list(self):
        # return a list of Detection objects, i.e. 'for result in results.tolist():'
        dets = []
        for *box, conf, class_id in self.pred:
            det = Detection(box, conf, class_id)
            dets.append(det)
        return dets

    def merge(self, other):
        # merge two Detections objects
        if isinstance(other, Detections):
            other = other.pred

        # check if other is empty
        if len(other) == 0:
            return self
        pred = np.concatenate((self.pred, other), axis=0)
        return Detections(pred, self.im, self.names, self.times)

    def __len__(self):
        return len(self.pred)


class BaseDetectionModel(ABC):
    def __init__(
        self, model_name: str, model_ckpt: str, inference_config: Dict[str, Any] = {}
    ):
        super().__init__()
        self.model_name = model_name
        self.model_ckpt = model_ckpt
        self.inference_config = inference_config
        self.model = self.load()

    def __call__(self, inputs, **kwargs):
        inputs = self._check_and_fix_inputs(inputs)
        results = self.forward(inputs, **kwargs)
        results = self._check_and_fix_outputs(results)
        detections = self._postprocess(results, inputs)
        return detections

    def _check_and_fix_inputs(self, inputs):
        """Check input type and shape.

        Acceptable input types are numpy.ndarray, torch.Tensor, pathlib Path, string file, PIL Image, or a list of any of these. All inputs will be converted to a list of numpy arrays.
        """
        if not isinstance(inputs, (list, tuple)):
            self.input_is_batched = False
            inputs = [inputs]
        else:
            self.input_is_batched = True

        imgs = []
        for img in inputs:
            img = read_image(img)
            imgs.append(img)
        return inputs

    def _check_and_fix_outputs(self, outputs):
        """Check output type and shape."""
        return outputs

    def _postprocess(self, outputs, inputs):
        """Postprocess the results."""
        detections = [Detections(o, i) for o, i in zip(outputs, inputs)]
        if not self.input_is_batched:
            return detections[0]

        return detections

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def forward(self, x):
        """
        Args:
            x (Tensor): input tensor
        Returns:
            Tensor: output tensor
        """
        raise NotImplementedError

    def test(self):
        import cv2
        import numpy as np
        from PIL import Image

        from ..utils.utils import get_git_root

        # batched inference
        git_root = get_git_root()
        # im_path = git_root / "data" / "samples" / "ney.jpeg"
        im_path = "/Users/agiats/Projects/soccernet_tracking/data/SoccerNet/tracking-2023/train/SNMOT-060/img1/000001.jpg"
        imgs = [
            str(im_path),  # filename
            im_path,  # Path
            "https://ultralytics.com/images/zidane.jpg",  # URI
            cv2.imread(str(im_path))[:, :, ::-1],  # OpenCV
            Image.open(str(im_path)),  # PIL
            np.zeros((320, 640, 3)),  # numpy
        ]

        results = self(imgs)
        print(results)

        for img in imgs:
            results = self(img)
            print(results)
