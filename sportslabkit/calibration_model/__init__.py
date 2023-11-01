import inspect

from sportslabkit.calibration_model.base import BaseCalibrationModel
from sportslabkit.calibration_model.dummy import DummyCalibrationModel
from sportslabkit.calibration_model.fld import SimpleContourCalibrator, FLDCalibrator
from sportslabkit.logger import logger


__all__ = ["BaseCalibrationModel", "load", "show_available_models", "DummyCalibrationModel", "LineBasedCalibrator"]


def inheritors(cls):
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


def show_available_models():
    """
    Print the names of all available BaseDetectionModel models.

    The models are subclasses of BaseDetectionModel. The names are printed as a list to the console.
    """
    print(sorted([cls.__name__ for cls in inheritors(BaseCalibrationModel)]))


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
    for cls in inheritors(BaseCalibrationModel):
        if model_name in [cls.__name__.lower(), cls.__name__]:
            # Filtering the model_config to only include keys that match the parameters of the target class
            config = {k.lower(): v for k, v in model_config.items()}
            return cls(**config)

    logger.warning(
        f"Model {model_name} not found. Available models: {[cls.__name__ for cls in inheritors(BaseCalibrationModel)]} (lowercase is allowed)"
    )


if __name__ == "__main__":
    for cls in inheritors(BaseCalibrationModel):
        print(cls.__name__)
