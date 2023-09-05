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

from sportslabkit.mot.base import MultiObjectTracker


def with_callbacks(func):
    """
    Decorator for wrapping methods that require callback invocations.

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
    def __init__(self, team_classifier):
        """
        Initialize TeamClassificationCallback.

        Args:
            team_classifier: A trained object responsible for classifying teams.
                            This object is generally loaded from a pickle file that
                            contains a trained scikit-learn Pipeline.
                - The object should have a `predict` method with the following specifications:
                    - predict(input_features: np.ndarray) -> np.ndarray
                        - Input: `input_features` is an ndarray of shape `(num_samples, num_features)`.
                                `num_samples` is the number of samples, and `num_features` is the
                                feature dimension for each sample.
                        - Output: An ndarray of shape `(num_samples,)` containing the predicted team IDs.
                                For a 2-class problem, it will contain integers like 0 or 1.
                - Example: If you're using an SVM-based classifier saved using pickle, this `predict`
                        method would take a feature vector and output the corresponding team IDs
                        (either 0 or 1 in a 2-class problem).

        Note:
            The `team_classifier` is expected to be a serialized object (e.g., pickle file)
            conforming to the above `predict` method specifications. It's commonly generated
            using scikit-learn and saved for future use.
        """
        super().__init__()
        self.team_classifier = team_classifier

    def on_track_sequence_start(self, tracker: MultiObjectTracker) -> None:
        """
        Method called at the start of a track sequence. It logs the number of tracklets
        currently active and terminated in the tracker.

        Args:
            tracker (MultiObjectTracker): The instance of the tracker.

        Notes:
            Logs the total number of tracklets including both alive and dead ones.
        """
        all_tracklets = tracker.alive_tracklets + tracker.dead_tracklets
        print(f"Tracklet started with {len(all_tracklets)} tracklets!")

    def on_track_sequence_end(self, tracker: MultiObjectTracker) -> None:
        """
        Method called at the end of a track sequence. During this phase, team classification
        is performed on each tracklet using the `team_classifier.predict`.

        Args:
            tracker (MultiObjectTracker): The instance of the tracker.

        Notes:
            - Team classification is applied to each tracklet.
            - An N-dimensional feature vector is extracted for each tracklet
            using `tracklet.get_observations("feature")`.
            - `team_classifier.predict` is used to classify the tracklet into a team ID
            (0 or 1 in a 2-class problem).
        """
        print("Applying team classification method...")
        all_tracklets = tracker.alive_tracklets + tracker.dead_tracklets
        for tracklet in all_tracklets:
            tracklet_features = tracklet.get_observations("feature")
            predict_team_id = stats.mode(
                self.team_classifier.predict(tracklet_features)
            )
            tracklet.team_id = predict_team_id.mode
        print(f"Tracklet ended with {len(all_tracklets)} tracklets!")
