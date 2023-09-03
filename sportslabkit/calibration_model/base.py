from abc import ABC, abstractmethod

import numpy as np
import torch
from PIL import Image

from sportslabkit.utils import read_image


class BaseCalibrationModel(ABC):
    """
    Base class for detection models. This class implements basic functionality for handling input and output data, and requires subclasses to implement model loading and forward pass functionality.

    Subclasses should override the 'load' and 'forward' methods. The 'load' method should handle loading the model from the specified repository and checkpoint, and 'forward' should define the forward pass of the model. Then add `ConfigTemplates` for your model to define the available configuration options.

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
        model_config (Optional[dict]): The configuration for the model.
        input_is_batched (bool): Whether the input is batched or not. This is set by the `_check_and_fix_inputs` method.
    """

    def __init__(self):
        """
        Initializes the base detection model.

        Args:
            model_config (Optional[dict]): The configuration for the model. This is optional and can be used to pass additional parameters to the model.
        """
        super().__init__()
        self.input_is_batched = False  # initialize the input_is_batched attribute

    def __call__(self, inputs, **kwargs):
        inputs = self._check_and_fix_inputs(inputs)
        results = self.forward(inputs, **kwargs)
        results = self._check_and_fix_outputs(results, inputs)
        detections = self._postprocess(results)
        return detections

    def _check_and_fix_inputs(self, img):
        """Check input type and shape.

        Acceptable input types are numpy.ndarray, torch.Tensor, pathlib Path, string file, PIL Image, or a list of any of these. All inputs will be converted to a list of numpy arrays.
        """
        # if isinstance(inputs, (list, tuple, np.ndarray, torch.Tensor)):
        #     self.input_is_batched = isinstance(inputs, (list, tuple)) or (hasattr(inputs, "ndim") and inputs.ndim == 4)
        #     if not self.input_is_batched:
        #         inputs = [inputs]
        # else:
        #     inputs = [inputs]

        # imgs = []
        # for img in inputs:
        #     img = self.read_image(img)
        #     imgs.append(img)
        return self.read_image(img)

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

        # # The output should be a list or np.ndarray of 3x3 homography matrices
        # if isinstance(outputs, (list, np.ndarray)):
        #     if len(outputs) == 0:
        #         raise ValueError("Output is empty.")
        #     if isinstance(outputs[0], (list, np.ndarray)):
        #         if len(outputs[0]) != 3 or len(outputs[0][0]) != 3:
        #             raise ValueError("Output should be a list of 3x3 homography matrices.")
        #     else:
        #         raise ValueError("Output should be a list of 3x3 homography matrices.")
        return outputs

    def _postprocess(self, outputs):
        """An empty post-processing method that does nothing. Override in subclasses for additional processing if needed."""
        return outputs

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

        from sportslabkit.utils.utils import get_git_root

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
    model = BaseCalibrationModel()
    model.test()
