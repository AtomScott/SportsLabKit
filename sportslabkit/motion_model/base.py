from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Type, Union

import numpy as np

from sportslabkit import Tracklet


class BaseMotionModel(ABC):
    """Abstract base class for motion models.

    This class defines a common interface for all motion models.
    Derived classes should implement the update, and predict methods. MotionModels are procedural and stateless. The state of tracklet is managed by the Tracklet class. The tracklet must have the required observations and states for the motion model to work. If the tracklet doesn't have the required observations or states, the motion model will raise an error and tell the user which observations or states are missing.
    """

    hparam_search_space: Dict[str, Type] = {}
    required_observation_types: List[str] = NotImplemented
    required_state_types: List[str] = NotImplemented

    def __init__(self, is_multi_target=False):
        """Initialize the MotionModel."""

        self.input_is_batched = False  # initialize the input_is_batched attribute
        self.name = self.__class__.__name__
        self.is_multi_target = is_multi_target

    def __call__(self, tracklet: Tracklet) -> Any:
        """Call the motion model to update its state and return the prediction.

        Args:
            tracklet (Tracklet): The single object tracker instance.

        Returns:
            Any: The predicted state after updating the motion model.
        """
        if self.is_multi_target:
            return self._multi_target_call(tracklet)

        self._check_required_observations(tracklet)
        self._check_required_states(tracklet)

        if isinstance(tracklet, Tracklet):
            _obs = tracklet.get_observations()
            observations = {t: _obs[t] for t in self.required_observation_types}
        else:
            observations = {t: tracklet[t] for t in self.required_observation_types}

        prediction, new_states = self.predict(observations, tracklet.states)
        tracklet.update_states(new_states)
        return prediction

    def _multi_target_call(self, tracklets: List[Tracklet]) -> List[Any]:
        """Call the motion model to update its state and return the prediction for multiple targets.

        Args:
            tracklets (List[Tracklet]): The list of tracklet instances.

        Returns:
            List[Any]: The list of predicted states after updating the motion model for each tracklet.
        """
        all_observations = []
        all_states = []
        for tracklet in tracklets:
            self._check_required_observations(tracklet)
            self._check_required_states(tracklet)

            if isinstance(tracklet, Tracklet):
                _obs = tracklet.get_observations()
                observations = {t: _obs[t] for t in self.required_observation_types}
            else:
                observations = {t: tracklet[t] for t in self.required_observation_types}
            all_observations.append(observations)
            all_states.append(tracklet.states)

        all_predictions, all_new_states = self.predict(all_observations, all_states)
        for i, tracklet in enumerate(tracklets):
            tracklet.update_states(all_new_states[i])
        return all_predictions

    @abstractmethod
    def predict(
        self,
        observations: Union[float, np.ndarray],
        states: Union[float, np.ndarray, None],
    ) -> Tuple[Union[float, np.ndarray, None], Union[float, np.ndarray]]:
        """Compute the next internal state and prediction based on the current observation and internal state.

        Args:
            observation (Union[float, np.ndarray]): The current observation.
            states (Union[float, np.ndarray, None]): The current internal state of the motion model.

        Returns:
            Tuple[Union[float, np.ndarray, None], Union[float, np.ndarray]]: The next internal state and the prediction.
        """
        pass

    @classmethod
    def from_config(cls: Type["BaseMotionModel"], config: Dict) -> "BaseMotionModel":
        """Initialize a motion model instance from a configuration dictionary.

        Args:
            config (Dict): The configuration dictionary containing the motion model's parameters.

        Returns:
            MotionModel: A new instance of the motion model initialized with the given configuration.
        """
        return cls(**config)

    def _check_required_observations(self, tracklet: Tracklet) -> None:
        """Check if the required observations are registered in the SingleObjectTracker instance.

        Args:
            sot (SingleObjectTracker): The single object tracker instance.

        Raises:
            KeyError: If a required observation is not registered in the SingleObjectTracker instance.
        """
        for obs_type in self.required_observation_types:
            if obs_type not in tracklet._observations:
                raise KeyError(f"{self.name} requires observation type `{obs_type}` but it is not registered.")
            if len(tracklet._observations[obs_type]) == 0:
                raise KeyError(f"{self.name} requires observation type `{obs_type}` but it is empty.")

    def _check_required_states(self, tracklet: Tracklet) -> None:
        """Check if the required states are registered in the SingleObjectTracker instance.

        Args:
            sot (SingleObjectTracker): The single object tracker instance.

        Raises:
            KeyError: If a required state is not registered in the SingleObjectTracker instance.
        """
        for state in self.required_state_types:
            if state not in tracklet._states:
                tracklet.register_state_type(state)
