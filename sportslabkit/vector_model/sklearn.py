from typing import Any

from joblib import load
from sklearn.pipeline import Pipeline

from sportslabkit.types import Vector
from sportslabkit.utils import fetch_or_cache_model
from sportslabkit.vector_model.base import BaseVectorModel


class SklearnVectorModel(BaseVectorModel):
    """
    A specialized subclass of BaseVectorModel for scikit-learn pipelines.

    This class is designed to facilitate the use of scikit-learn pipelines as vector-based models
    within the SportsLabKit ecosystem. It overrides the abstract methods from BaseVectorModel
    to provide implementations tailored for scikit-learn pipelines.

    Attributes:
        model (Pipeline | None): The loaded scikit-learn pipeline model. None if the model is not loaded.
    """
    def __init__(
        self,
        model_path: str = "",
        input_vector_size: int | None = None, 
        output_vector_size: int | None = None) -> None:
        super().__init__(input_vector_size, output_vector_size)
        self.model_path = model_path
        self.load(model_path)
        

    def forward(self, inputs: Vector, **kwargs: Any) -> Vector:
        """
        Implement the forward pass specific to scikit-learn pipelines.

        This method takes a vector input and passes it through the scikit-learn pipeline's
        `predict` method. Additional keyword arguments can be passed to the `predict` method
        via **kwargs.

        Args:
            inputs (Vector): The input vector, which should match the expected input shape of the pipeline.
            **kwargs (Any): Additional keyword arguments to pass to the pipeline's `predict` method.

        Returns:
            Vector: The output vector from the pipeline's `predict` method.

        Raises:
            ValueError: If the model attribute is None, indicating that the model has not been loaded.
        """
        if self.model is None:
            raise ValueError("The model is as empty as a politician's promise. Load it first.")

        return self.model.predict(inputs, **kwargs)[0]

    def _load_model(self, path: str) -> None:
        """
        Load a scikit-learn pipeline model from disk using joblib.

        This method uses joblib to load a pre-trained scikit-learn pipeline from the specified file path.
        The loaded model is stored in the `model` attribute. A type check is performed to ensure
        that the loaded object is a scikit-learn pipeline.

        Args:
            path (str): The file path to the pre-trained scikit-learn pipeline.

        Raises:
            TypeError: If the loaded model is not a scikit-learn pipeline.
        """
        actual_path = fetch_or_cache_model(path)
        self.model = load(actual_path)

        if not isinstance(self.model, Pipeline):
            raise TypeError(f"Oops, you loaded something that's not a pipeline. Got a {type(self.model)} instead.")
