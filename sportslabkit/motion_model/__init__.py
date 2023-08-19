import inspect
from sportslabkit.motion_model.tune import tune_motion_model
from sportslabkit.motion_model.models import ExponentialMovingAverage, KalmanFilter
from sportslabkit.motion_model.base import BaseMotionModule, BaseMotionModel
from sportslabkit.logger import logger

__all__ = ["tune_motion_model", "ExponentialMovingAverage", "KalmanFilter", "BaseMotionModule"]


def inheritors(cls: type) -> set[type]:
    """
    Get all subclasses of a given class.

    Args:
        cls (type): The class to find subclasses of.

    Returns:
        set[type]: A set of the subclasses of the input class.
    """
    subclasses = set()
    work = [cls]
    while work:
        parent = work.pop()
        for child in parent.__subclasses__():
            if child not in subclasses:
                subclasses.add(child)
                work.append(child)
    return subclasses


def show_available_models() -> None:
    """
    Print the names of all available BaseDetectionModel models.

    The models are subclasses of BaseDetectionModel. The names are printed as a list to the console.
    """
    print(sorted([cls.__name__ for cls in inheritors(BaseMotionModel)]))


def load(model_name, **model_config):
    """
    Load a model by name.

    The function searches subclasses of BaseDetectionModel for a match with the given name. If a match is found, an instance of the model is returned. If no match is found, a warning is logged and the function returns None.

    Args:
        model_name (str): The name of the model to load.
        model_config (dict, optional): The model configuration to use when instantiating the model.

    Returns:
        BaseDetectionModel: An instance of the requested model, or None if no match was found.
    """
    for cls in inheritors(BaseMotionModel):
        if model_name in [cls.__name__.lower(), cls.__name__]:
            # Filtering the model_config to only include keys that match the parameters of the target class
            filtered_config = {k.lower(): v for k, v in model_config.items() if k.lower() in inspect.signature(cls.__init__).parameters}
            return cls(**filtered_config)

    logger.warning(f"Model {model_name} not found. Available models: {[cls.__name__ for cls in inheritors(BaseMotionModel)]} (lowercase is allowed)")


if __name__ == "__main__":
    for cls in inheritors(BaseMotionModel):
        print(cls.__name__)
