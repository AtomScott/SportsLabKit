from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from PIL import Image
from torch import nn
from torchvision import transforms

from soccertrack.image_model.visualization import plot_tsne
from soccertrack.logger import logger
from soccertrack.types.detection import Detection
from soccertrack.types.detections import Detections
from soccertrack.utils import read_image


@dataclass
class BaseConfig:
    """Base class for configuration dataclasses.

    This class implements basic functionality for handling configuration dataclasses. It requires subclasses to implement a `from_dict` method that converts a dictionary to the dataclass.

    Example:
        @dataclass
        class MyConfig(BaseConfig):
            my_field: str
            my_other_field: int

            @classmethod
            def from_dict(cls, config: dict):
                return cls(**config)

        config = MyConfig.from_dict({"my_field": "foo", "my_other_field": 42})
    """

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        return cls(**data)


def validate_config(config: dict, config_class):
    """
    Validate a configuration dictionary against a dataclass.

    Args:
        config (dict): The configuration dictionary to validate.
        config_class (type): The dataclass to validate against.

    Returns:
        dict: The updated configuration dictionary, containing the default options updated/overwritten by those passed in the config argument.
    """
    # Get the names of all fields in the dataclass
    valid_keys = {f.name for f in fields(config_class)}
    # Check if there are any unknown keys in the config
    unknown_keys = set(config.keys()) - valid_keys
    if unknown_keys:
        raise ValueError(
            f"Unknown keys in configuration: {unknown_keys}. Valid keys are {valid_keys}"
        )

    # Create a dictionary with the default options of the dataclass
    default_config = asdict(config_class())
    # Update the default configuration with the input configuration
    default_config.update(config)

    try:
        # Try to create an instance of the dataclass
        config_class(**default_config)
    except TypeError as e:
        # If a TypeError was raised, an argument was missing or had the wrong type
        missing_keys = valid_keys - set(default_config.keys())
        raise ValueError(
            f"Missing or incorrect type keys in configuration: {missing_keys}. Error: {e}"
        )

    logger.debug(f"Configuration: {default_config}")

    return default_config


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

    def __init__(self, model_config={}, inference_config={}):
        """
        Initializes the base image embedding model.

        Args:
            model_config (Optional[dict]): The configuration for the model. This is optional and can be used to pass additional parameters to the model.
            inference_config (Optional[dict]): The configuration for the inference. This is optional and can be used to pass additional parameters to the inference.
        """
        super().__init__()
        self.model_config = self.validate_model_config(model_config)
        self.inference_config = self.validate_inference_config(inference_config)
        self.input_is_batched = False  # initialize the input_is_batched attribute

        self.model = self.load()

    def __call__(self, inputs, **kwargs):
        inputs = self._check_and_fix_inputs(inputs)
        results = self.forward(inputs, **kwargs)
        embeddings = self._check_and_fix_outputs(results)
        return embeddings

    def _check_and_fix_inputs(self, inputs) -> List[np.ndarray]:
        """Check input type and shape.

        Acceptable input types are numpy.ndarray, torch.Tensor, pathlib Path, string file, PIL Image, or a list of any of these. All inputs will be converted to a list of numpy arrays.
        """
        if isinstance(inputs, (list, tuple, np.ndarray, torch.Tensor)):
            self.input_is_batched = isinstance(inputs, (list, tuple)) or (
                hasattr(inputs, "ndim") and inputs.ndim == 4
            )
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

        # convert the output to a 2D numpy array
        return np.stack(outputs)

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
    def forward(self, x: np.ndarray):
        """
        Forward must be overridden by subclasses. The overriding method should define the forward pass of the model. The model will receive a 4-dimensional numpy array representing the images. As for the output, the model is expected to return something that can be converted into a 2-dimensional numpy array.

        Args:
            x (np.ndarray): input image
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def model_config_template(self) -> BaseConfig:
        # return a subclass of BaseConfig or some kind of dataclass
        raise NotImplementedError

    @property
    @abstractmethod
    def inference_config_template(self) -> BaseConfig:
        # return a subclass of BaseConfig or some kind of dataclass
        raise NotImplementedError

    def show_model_config(self):
        logger.info("Model configuration:")
        for key, value in self.model_config.items():
            logger.info(f"  {key}: {value}")

    def show_inference_config(self):
        logger.info("Inference configuration:")
        for key, value in self.inference_config.items():
            logger.info(f"  {key}: {value}")

    def validate_model_config(self, config: dict):
        return validate_config(config, self.model_config_template)

    def validate_inference_config(self, config: dict):
        return validate_config(config, self.inference_config_template)

    def embed_detections(
        self, detections: Sequence[Detection], image: Union[Image.Image, np.ndarray]
    ) -> np.ndarray:
        if isinstance(detections, Detections):
            detections = detections.to_list()

        if isinstance(image, Image.Image):
            image = np.array(image)

        box_images = []
        for detection in detections:
            x, y, w, h = list(map(int, detection.box))
            box_image = image[y : y + h, x : x + w]
            box_images.append(box_image)

        with torch.no_grad():
            z = self(box_images)

        return z
