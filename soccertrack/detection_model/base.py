from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import requests
from PIL import Image

from soccertrack.logger import logger
from soccertrack.types import Detection
from soccertrack.types.detection import Detection
from soccertrack.types.detections import Detections
from soccertrack.utils import read_image
from soccertrack.utils.draw import draw_bounding_boxes


def convert_to_detection(pred):
    """Convert an output to a single detection object.

    Handles the following input types:
    - dict with keys: bbox_left, bbox_top, bbox_width, bbox_height, conf, class
    - list or tuple with 6 items: bbox_left, bbox_top, bbox_width, bbox_height, conf, class
    - Detection object

    Args:
        pred: prediction object to convert

    Returns:
        Detection object
    """

    if isinstance(pred, dict):
        if len(pred.keys()) != 6:
            raise ValueError("The prediction dictionary should contain exactly 6 items")
        return Detection(
            box=np.array(
                [
                    pred["bbox_left"],
                    pred["bbox_top"],
                    pred["bbox_width"],
                    pred["bbox_height"],
                ]
            ),
            score=pred["conf"],
            class_id=pred["class"],
        )

    elif (
        isinstance(pred, list)
        or isinstance(pred, tuple)
        or isinstance(pred, np.ndarray)
    ):
        if len(pred) != 6:
            raise ValueError("The prediction list should contain exactly 6 items")
        return Detection(box=np.array(pred[:4]), score=pred[4], class_id=pred[5])
    elif isinstance(pred, Detection):
        return pred
    else:
        raise TypeError(f"Unsupported prediction type: {type(pred)}")


class BaseDetectionModel(ABC):
    """
    Base class for detection models. This class implements basic functionality for handling input and output data, and requires subclasses to implement model loading and forward pass functionality.

    Subclasses should override the 'load' and 'forward' methods. The 'load' method should handle loading the model from the specified repository and checkpoint, and 'forward' should define the forward pass of the model.

    The input to the model should be flexible. It accepts numpy.ndarray, torch.Tensor, pathlib Path, string file, PIL Image, or a list of any of these. All inputs will be converted to a list of numpy arrays representing the images.

    The output of the model is expected to be a list of `Detection` objects, where each `Detection` object represents a detected object in an image. If the model's output does not meet this expectation, `_check_and_fix_outputs` method should convert the output into a compatible format.

    Example:
        class CustomDetectionModel(BaseDetectionModel):
            def load(self):
                # Load your model here
                pass

            def forward(self, x):
                # Define the forward pass here
                pass

    Attributes:
        model_name (str): The name of the model.
        model_repo (str): The repository where the model is stored.
        model_ckpt (str): The checkpoint of the model to load.
        model_config (Optional[dict]): The configuration for the model.
        input_is_batched (bool): Whether the input is batched or not.
        model (Any): The loaded model.
    """

    def __init__(self, model_name, model_repo, model_ckpt, model_config=None):
        """
        Initializes the base detection model.

        Args:
            model_name (str): The name of the model.
            model_repo (str): The repository where the model is stored.
            model_ckpt (str): The checkpoint of the model to load.
            model_config (Optional[dict]): The configuration for the model.
        """
        super().__init__()
        self.model_name = model_name
        self.model_ckpt = model_ckpt
        self.model_config = model_config
        self.input_is_batched = False  # initialize the input_is_batched attribute

        self.model = self.load()

    def __call__(self, inputs, **kwargs):
        inputs = self._check_and_fix_inputs(inputs)
        results = self.forward(inputs, **kwargs)
        results = self._check_and_fix_outputs(results, inputs)
        detections = self._postprocess(results)
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
            img = self.read_image(img)
            imgs.append(img)
        return imgs

    def read_image(self, img):
        return read_image(img)

    def _check_and_fix_outputs(self, outputs, inputs):
        """
        Check output type and convert to list of `Detections` objects.

        The function expects the raw output from the model to be either a list of `Detection` objects or a list of lists, where each sub-list should contain four elements corresponding to the bounding box of the detected object. See `Detection` and `Detections` class for more details.

        If the output is not in the correct format, a ValueError is raised.

        Args:
            outputs: The raw output from the model.
            inputs: The corresponding inputs to the model.

        Returns:
            A list of `Detections` objects.
        """
        # First the length of outputs and inputs should be equal.
        if len(outputs) != len(inputs):
            raise ValueError(
                "Length of outputs does not match length of inputs. "
                f"Got {len(outputs)} outputs and {len(inputs)} inputs."
            )

        if isinstance(outputs[0], Detections):
            return outputs

        # A common mistake is that the model returns a single Detection object instead of a list of Detection objects, especially when the model is inferring a single image.
        check_1 = not isinstance(outputs, (list, tuple)) or not isinstance(
            outputs[0], (list, tuple)
        )
        check_2 = isinstance(outputs[0][0], int) or isinstance(outputs[0][0], float)
        if check_1 or check_2:
            raise ValueError(
                "The model's output should be a list of list of Detection objects or a compatible object."
            )

        # Attempt to convert outputs into a list of Detections objects.
        list_of_detections = []
        for preds, image in zip(outputs, inputs):
            dets = []
            for pred in preds:
                if pred == {}:
                    continue
                det = convert_to_detection(pred)
                dets.append(det)

            list_of_detections.append(Detections(dets, image))

        if not list_of_detections:
            raise ValueError("Empty list of detections. Check your model's output.")
        # Raise ValueError if lengths of detections and inputs are not equal.
        if len(list_of_detections) != len(inputs):
            raise ValueError("Length of detections does not match length of inputs.")
        return list_of_detections

    def _postprocess(self, outputs):
        """An empty post-processing method that does nothing. Override in subclasses for additional processing if needed."""
        return outputs

    @abstractmethod
    def load(self):
        """
        Loads the model.

        This method must be overridden by subclasses. The overriding method should load the model from the repository using the checkpoint, and return the loaded model.

        Raises:
            NotImplementedError: If the method is not overridden.
        """
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

        from soccertrack.utils.utils import get_git_root

        # batched inference
        git_root = get_git_root()
        im_path = git_root / "data" / "samples" / "ney.jpeg"
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


if __name__ == "__main__":
    model = BaseDetectionModel("model_name", "model_repo", "model_ckpt")
    model.test()
