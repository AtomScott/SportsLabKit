"""Defines the Callback base class and utility decorators for use with the Trainer class.

The Callback class provides a dynamic way to hook into various stages of the Trainer's operations.
It uses Python's __getattr__ method to dynamically handle calls to methods that are not explicitly defined,
allowing it to handle arbitrary `on_<event_name>_start` and `on_<event_name>_end` methods.

Example:
    class MyPrintingCallback(Callback):
        def on_train_start(self, trainer):
            print("Training is starting")
"""

from functools import wraps

from scipy import stats
from sportskit.logger import logger

from sportslabkit.mot.base import MultiObjectTracker
from sportslabkit.types import Vector
from sportslabkit.vector_model import BaseVectorModel


def with_callbacks(func):
    """Decorator for wrapping methods that require callback invocations.

    Args:
        func (callable): The method to wrap.

    Returns:
        callable: The wrapped method.
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        event_name = func.__name__
        self._invoke_callbacks(f"on_{event_name}_start")
        result = func(self, *args, **kwargs)
        self._invoke_callbacks(f"on_{event_name}_end")
        return result

    return wrapper


class Callback:
    """Base class for creating new callbacks.

    This class defines the basic structure of a callback and allows for dynamic method creation
    for handling different events in the Trainer's lifecycle.

    Methods:
        __getattr__(name: str) -> callable:
            Returns a dynamically created method based on the given name.
    """

    pass


class TeamClassificationCallback(Callback):
    def __init__(self, vector_model: BaseVectorModel):
        """Initialize TeamClassificationCallback.

        Args:
            vector_model (BaseVectorModel): A trained object responsible for classifying teams.
        """
        super().__init__()
        self.vector_model = vector_model

    def on_track_sequence_end(self, tracker: MultiObjectTracker) -> None:
        """Method called at the end of a track sequence. Applies team classification.

        Args:
            tracker (MultiObjectTracker): The instance of the tracker.
        """
        logger.debug("Applying team classification method...")
        all_tracklets = tracker.alive_tracklets + tracker.dead_tracklets

        for tracklet in all_tracklets:
            tracklet_features: Vector = tracklet.get_observations("feature")

            # Using forward method for model inference
            predicted_team_id = self.vector_model(tracklet_features)

            # Assuming you want the most frequent prediction as the final team ID
            most_frequent_team_id = stats.mode(predicted_team_id).mode[0]
            tracklet.team_id = most_frequent_team_id
