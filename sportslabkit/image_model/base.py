from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

import numpy as np
import torch
from PIL import Image

from sportslabkit.types.detection import Detection
from sportslabkit.types.detections import Detections
from sportslabkit.utils import read_image


class BaseImageModel(ABC):
    """
    Base class for image embedding models. This class implements basic functionality for handling input and output data, and requires subclasses to implement model loading and forward pass functionality.

    Subclasses should override the 'load' and 'forward' methods. The 'load' method should handle loading the model from the specified repository and checkpoint, and 'forward' should define the forward pass of the model. Then add `ConfigTemplates` for your model to define the available configuration options.

    The input to the model should be flexible. It accepts numpy.ndarray, torch.Tensor, pathlib Path, string file, PIL Image, or a list of any of these. All inputs will be converted to a list of numpy arrays representing the images.

    The output of the model is expected to be a list of embeddings, one for each image. If the model's output does not meet this expectation, `_check_and_fix_outputs` method should convert the output into a compatible format.

    Example:
        class CustomImageModel(BaseImageModel):
            def load(self):
                # Load your model here
                pass

            def forward(self, x):
                # Define the forward pass here
                pass

    Attributes:
        model_config (Optional[dict]): The configuration for the model.
        inference_config (Optional[dict]): The configuration for the inference.
        input_is_batched (bool): Whether the input is batched or not. This is set by the `_check_and_fix_inputs` method.
    """

    def __init__(self):
        """
        Initializes the base image embedding model.

        Args:
            model_config (Optional[dict]): The configuration for the model. This is optional and can be used to pass additional parameters to the model.
            inference_config (Optional[dict]): The configuration for the inference. This is optional and can be used to pass additional parameters to the inference.
        """
        super().__init__()
        self.input_is_batched = False  # initialize the input_is_batched attribute

    def __call__(self, inputs, **kwargs):
        inputs = self._check_and_fix_inputs(inputs)
        results = self.forward(inputs, **kwargs)
        embeddings = self._check_and_fix_outputs(results)
        return embeddings

    def _check_and_fix_inputs(self, inputs) -> list[np.ndarray]:
        """Check input type and shape.

        Acceptable input types are numpy.ndarray, torch.Tensor, pathlib Path, string file, PIL Image, or a list of any of these. All inputs will be converted to a list of numpy arrays.
        """
        if isinstance(inputs, (list, tuple, np.ndarray, torch.Tensor)):
            self.input_is_batched = isinstance(inputs, (list, tuple)) or (hasattr(inputs, "ndim") and inputs.ndim == 4)
            if not self.input_is_batched:
                inputs = [inputs]
        else:
            inputs = [inputs]

        imgs = []
        for img in inputs:
            img = self.read_image(img)
            imgs.append(img)

        return imgs

    def read_image(self, img):
        return read_image(img)

    def _check_and_fix_outputs(self, outputs):
        """
        Check output type and convert to a 2D numpy array.

        The function expects the raw output from the model to be either an iterable of embeddings of the same length as the input, or a single embedding. If the output is not in the correct format, a ValueError is raised.

        If the output is not in the correct format, a ValueError is raised.

        Args:
            outputs: The raw output from the model.

        Returns:
            A list of embeddings.
        """
        # Check if the output is an iterable
        if not isinstance(outputs, Iterable):
            raise ValueError(f"Model output is not iterable. Got {type(outputs)}")

        # Check if the first element in the list is a list or a single value.
        # If it's a single value, each output is a single embedding.
        # If it's a list, each output is a list of embeddings.
        if isinstance(outputs[0], (int, float)):
            outputs = [[output] for output in outputs]

        # Check if the array is torch.Tensor or numpy.ndarray
        if isinstance(outputs[0], torch.Tensor):
            outputs = [output.detach().cpu().numpy() for output in outputs]

        # convert the output to a 2D numpy array
        return np.stack(outputs)

    @abstractmethod
    def forward(self, x: np.ndarray):
        """
        Forward must be overridden by subclasses. The overriding method should define the forward pass of the model. The model will receive a 4-dimensional numpy array representing the images. As for the output, the model is expected to return something that can be converted into a 2-dimensional numpy array.

        Args:
            x (np.ndarray): input image
        """
        raise NotImplementedError

    def embed_detections(self, detections: Sequence[Detection], image: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(detections, Detections):
            detections = detections.to_list()

        if isinstance(image, Image.Image):
            image = np.array(image)

        box_images = []
        for detection in detections:
            if isinstance(detection, Detection):
                x, y, w, h = list(map(int, detection.box))
            elif isinstance(detection, dict):
                x, y, w, h = list(map(int, detection["box"]))
            else:
                raise ValueError(f"Unknown detection type: {type(detection)}")
            box_image = image[y : y + h, x : x + w]
            box_images.append(box_image)

        with torch.no_grad():
            z = self(box_images)

        return z
