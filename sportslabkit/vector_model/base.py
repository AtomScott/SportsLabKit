from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch

from sportslabkit.types import Vector


class BaseVectorModel(ABC):
    """
    Abstract Base Class for handling vector-based models.
    This class encapsulates model loading, input/output validation, and forward pass operations.
    """

    def __init__(self, input_vector_size: int | None = None, output_vector_size: int | None = None) -> None:
        """
        Initialize the BaseVectorModel.

        Args:
            input_vector_size (Optional[int]): The size of the input vector. None to bypass validation.
            output_vector_size (Optional[int]): The size of the output vector. None to bypass validation.
        """
        super().__init__()
        self.input_vector_size = input_vector_size
        self.output_vector_size = output_vector_size
        self.model = None  # Placeholder for the actual model

    def __call__(self, inputs: Vector, **kwargs: Any) -> Vector:
        """
        Call the model's forward method after input validation and before output validation.

        Args:
            inputs (Vector): The input data.
            **kwargs (Any): Additional arguments to be passed to the forward method.

        Returns:
            Vector: The output data.
        """
        inputs = self._check_and_fix_inputs(inputs)
        outputs = self.forward(inputs, **kwargs)
        return self._check_and_fix_outputs(outputs)

    def _check_and_fix_inputs(self, inputs: Vector) -> Vector:
        """
        Validate and optionally fix the inputs before feeding them to the model.

        Args:
            inputs (Vector): The input data.

        Returns:
            Vector: The validated and possibly fixed input data.
        """
        if self.input_vector_size and len(inputs) != self.input_vector_size:
            raise ValueError(f"Input vector size mismatch. Expected {self.input_vector_size}, got {len(inputs)}.")

        return np.array(inputs) if isinstance(inputs, list) else inputs

    def _check_and_fix_outputs(self, outputs: Vector) -> Vector:
        """
        Validate and optionally fix the outputs before returning them.

        Args:
            outputs (Vector): The output data.

        Returns:
            Vector: The validated and possibly fixed output data.
        """
        if self.output_vector_size and len(outputs) != self.output_vector_size:
            raise ValueError(f"Output vector size mismatch. Expected {self.output_vector_size}, got {len(outputs)}.")

        return np.array(outputs) if isinstance(outputs, torch.Tensor) else outputs

    @abstractmethod
    def forward(self, inputs: Vector, **kwargs: Any) -> Vector:
        """
        Define the forward pass of the model. Must be overridden by subclasses.

        Args:
            inputs (Vector): The input data.
            **kwargs (Any): Additional arguments to be passed to the forward method.

        Returns:
            Vector: The output data.
        """
        raise NotImplementedError("The forward method must be implemented by subclasses.")

    def load(self, path: str) -> None:
        """
        Load the model from disk.

        Args:
            path (str): The path to the model file.
        """
        self._load_model(path)
        self._post_load_check()

    @abstractmethod
    def _load_model(self, path: str) -> None:
        """
        The actual model loading logic. Must be overridden by subclasses.

        Args:
            path (str): The path to the model file.
        """
        raise NotImplementedError("The _load_model method must be implemented by subclasses.")

    def _post_load_check(self) -> None:
        """
        Check whether the model has been loaded correctly.
        """
        if self.model is None:
            raise ValueError("Model not loaded correctly. Fix your _load_model implementation.")
