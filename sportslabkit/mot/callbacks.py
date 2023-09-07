"""Defines the Callback base class and utility decorators for use with the Trainer class.

The Callback class provides a dynamic way to hook into various stages of the Trainer's operations.
It uses Python's __getattr__ method to dynamically handle calls to methods that are not explicitly defined,
allowing it to handle arbitrary `on_<event_name>_start` and `on_<event_name>_end` methods.

Example:
    class MyPrintingCallback(Callback):
        def on_train_start(self, trainer):
            print("Training is starting")
"""


from scipy import stats

from sportslabkit.logger import logger
from sportslabkit.mot.base import Callback, MultiObjectTracker
from sportslabkit.types import Vector
from sportslabkit.vector_model import BaseVectorModel


class TeamClassificationCallback(Callback):
    def __init__(self, vector_model: BaseVectorModel):
        """Initialize TeamClassificationCallback.

        Args:
            vector_model (BaseVectorModel): A trained object responsible for classifying teams.
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
            The `vector_model` is expected to be a serialized object (e.g., pickle file)
            conforming to the above `predict` method specifications. It's commonly generated
            using scikit-learn and saved for future use.
        """
        super().__init__()
        self.vector_model = vector_model

    def on_track_sequence_end(self, tracker: MultiObjectTracker) -> None:
        """Call the `vector_model.predict` method on each tracklet to classify it into a team ID.

        Method called at the end of a track sequence. During this phase, team classification
        is performed on each tracklet using the `vector_model.predict`.

        Args:
            tracker (MultiObjectTracker): The instance of the tracker.

        Notes:
            - Team classification is applied to each tracklet.
            - An N-dimensional feature vector is extracted for each tracklet
            using `tracklet.get_observations(“feature”)`.
            - `vector_model.predict` is used to classify the tracklet into a team ID
            (0 or 1 in a 2-class problem).
        """

        logger.debug("Applying team classification method...")
        all_tracklets = tracker.alive_tracklets + tracker.dead_tracklets

        for tracklet in all_tracklets:
            tracklet_features: Vector = tracklet.get_observations("feature")

            # Using forward method for model inference
            predicted_team_id = self.vector_model(tracklet_features)

            # Assuming you want the most frequent prediction as the final team ID
            most_frequent_team_id = stats.mode(predicted_team_id, axis=0, keepdims=False).mode
            tracklet.team_id = most_frequent_team_id
