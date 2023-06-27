from __future__ import annotations

import uuid
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd

from soccertrack.dataframe.bboxdataframe import BBoxDataFrame
from soccertrack.logger import logger
from soccertrack.types.detection import Detection
import hashlib


def id_to_color(id_string: str) -> str:
    hash_object = hashlib.md5(id_string.encode())
    hash_int = int(hash_object.hexdigest(), 16) % 12
    colors = [
        "\033[91m",  # Red
        "\033[92m",  # Green
        "\033[93m",  # Yellow
        "\033[94m",  # Blue
        "\033[95m",  # Magenta
        "\033[96m",  # Cyan
        "\033[97m",  # White
        "\033[31m",  # Bright Red
        "\033[32m",  # Bright Green
        "\033[33m",  # Bright Yellow
        "\033[34m",  # Bright Blue
        "\033[35m",  # Bright Magenta
    ]
    return colors[hash_int]


class Tracklet:
    """Tracklet class to be u

    Stores observations of different types without making predictions about the next state.
    New observation types can be registered, and the tracker can be extended with more functionality if needed.

    Observations are stored in a dictionary, where the key is the name of the observation type and the value is a list of observations. The length of the list is equal to the number of steps the tracker has been alive. The first element of the list is the first observation, and the last element is the most recent observation.

    States are stored in a dictionary, where the key is the name of the state and the value is the most recent state. The state is an indication of the current state of the tracker.

    Attributes:
        id (int): unique id of the tracker
        steps_alive (int): number of steps the tracker was alive
        steps_positive (int): number of steps the tracker was positive (i.e., had a detection associated with it)
        staleness (float): number of steps since the last positive update
        global_step (int): number of steps since the start of the tracking process
        max_staleness (float): number of steps after which a tracker is considered stale
    """

    def __init__(self):
        self.id: int = int(str(int(uuid.uuid4()))[:12])
        self.steps_alive: int = 0
        self.steps_positive: int = 0
        self.staleness: float = 0.0
        self.global_step: int = 0
        self.max_staleness: float = 12.0
        self._observations: Dict[str, List[Any]] = {}
        self._states: Dict[str, Any] = {}

        # # TODO: this shouldn't be hardcoded
        # for name in ["box", "score", "class_id", "feature"]:
        #     self.register_observation_type(name)

    def __getattr__(self, name: str) -> Any:
        if name in self._observations:
            return self.get_observation(name)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")

    def __repr__(self) -> str:
        return f"Tracklet(id={self.id}, current_box={self.box})"

    def register_observation_type(self, name: str) -> None:
        """Register a new observation type.

        Args:
            name (str): Name of the new observation type to be registered.
        """
        if name not in self._observations:
            self._observations[name] = []

    def register_observation_types(self, names: List[str]) -> None:
        """Register a new observation type.

        Args:
            name (str): Name of the new observation type to be registered.
        """
        for name in names:
            self.register_observation_type(name)

    def update_observations(self, observations: Dict[str, Any], global_step: Optional[int] = None) -> None:
        for name, value in observations.items():
            self.update_observation(name, value)

    def update_observation(self, name: str, value: Any, global_step: Optional[int] = None) -> None:
        if name in self._observations:
            self._observations[name].append(value)
        else:
            raise ValueError(f"Observation type '{name}' not registered")

    def register_state_type(self, name: str) -> None:
        """Register a new state type.

        Args:
            name (str): Name of the new state type to be registered.
        """
        if name not in self._states:
            self._states[name] = None

    def register_state_types(self, names: List[str]) -> None:
        """Register a new state type.

        Args:
            name (str): Name of the new state type to be registered.
        """
        for name in names:
            self.register_state_type(name)

    def get_observation(self, name: Optional[str] = None) -> Optional[Any]:
        """Get the most recent value of an observation type.

        Args:
            name (str): Name of the observation type.

        Returns:
            Optional[Any]: The most recent value of the specified observation type or None if not available.
        """
        if name is None:
            return [self._observations[name][-1] for name in self._observations]
        if name in self._observations and self._observations[name]:
            return self._observations[name][-1]
        return None

    def get_observations(self, name: Optional[str] = None) -> Optional[Any]:
        """Get all values of an observation type.

        Args:
            name (str): Name of the observation type.

        Returns:
            List[Any]: All values of the specified observation type.
        """
        if name is None:
            return self._observations
        if name in self._observations:
            return self._observations[name]
        else:
            raise ValueError(f"Observation type '{name}' not registered")

    def get_state(self, name: Optional[str] = None) -> Optional[Any]:
        """Get the most recent value of a state type.

        Args:
            name (str): Name of the state type.

        Returns:
            Optional[Any]: The most recent value of the specified state type or None if not available.
        """
        if name is None:
            return [self._states[name] for name in self._states]
        if name in self._states:
            return self._states[name]
        return None

    def get_states(self, name: Optional[str] = None) -> Optional[Any]:
        """Get all values of a state type.

        Args:
            name (str): Name of the state type.

        Returns:
            List[Any]: All values of the specified state type.
        """
        if name is None:
            return self._states
        if name in self._states:
            return self._states[name]
        else:
            raise ValueError(f"State type '{name}' not registered")

    def update_states(self, states: Dict[str, Any], global_step: Optional[int] = None) -> None:
        """Update multiple states with new values.

        Args:
            states (Dict[str, Any]): Dictionary of states to be updated.
            global_step (Optional[int], optional): Global step. Defaults to None.
        """
        for name, value in states.items():
            self.update_state(name, value)

    def update_state(self, name: str, value: Any) -> None:
        """Update the state with a new value.

        Args:
            name (str): Name of the state to be updated.
            value (Any): New value for the specified state.
        """
        if name in self._states:
            self._states[name] = value
        else:
            raise ValueError(f"State type '{name}' not registered")

    # FIXME: Maybe refactor this to be override_current_observation?
    def update_current_observation(self, name: str, value: Any) -> None:
        """Update the most recent observation with a new value.

        Args:
            name (str): Name of the observation type to be updated.
            value (Any): New value for the specified observation type.
        """
        if name in self._observations:
            self._observations[name][-1] = value
        else:
            raise ValueError(f"Observation type '{name}' not registered")

    def increment_counter(self, global_step: Optional[int] = None) -> None:
        self.steps_alive += 1
        if global_step is not None:
            self.global_step = int(global_step)
        else:
            self.global_step += 1

    def is_active(self) -> bool:
        """Check if the tracker is active.

        Returns:
            bool: True if the tracker is active (i.e. steps_alive > 0, not stale, and not invalid), otherwise False.
        """
        return self.steps_alive > 0 and not self.is_stale()

    def is_stale(self) -> bool:
        """Check if the tracker is stale.

        Returns:
            bool: True if the tracker's staleness is greater than max_staleness, otherwise False.
        """
        return self.staleness > self.max_staleness

    def to_bbdf(self) -> BBoxDataFrame:
        """Convert the tracker predictions to a BBoxDataFrame.

        Returns:
            BBoxDataFrame: BBoxDataFrame of the tracker
        """

        if len(self.box) == 0:
            return pd.DataFrame()

        if self.global_step >= self.steps_alive:
            frame_range = range(self.global_step + 1 - self.steps_alive, self.global_step + 1)
        else:
            raise ValueError(f"Global step {self.global_step} is less than steps alive {self.steps_alive}")

        data_dict = {"frame": list(frame_range), "id": [self.id for _ in frame_range]}
        for observation in self._observations:
            if self.get_observation(observation) is not None:
                data_dict[observation] = self.get_observations(observation)
            elif observation == "score":
                data_dict[observation] = [1 for _ in frame_range]

        df = pd.DataFrame(data_dict)

        df = pd.DataFrame(df["box"].to_list(), columns=["bb_left", "bb_top", "bb_width", "bb_height"]).join(
            df.drop(columns=["box"])
        )

        df.rename(columns={"global_step": "frame", "score": "conf"}, inplace=True)

        df.set_index(["frame"], inplace=True)

        box_df = df[["bb_left", "bb_top", "bb_width", "bb_height", "conf"]]
        team_id = 0
        player_id = df.id.unique()[0]

        idx = pd.MultiIndex.from_product(
            [[team_id], [player_id], box_df.columns],
            names=["TeamID", "PlayerID", "Attributes"],
        )

        bbdf = BBoxDataFrame(box_df.values, index=df.index, columns=idx)
        return bbdf

    # def update(
    #     self,
    #     detection: Union[Detection, None],
    #     states: Optional[Dict[str, Any]] = None,
    # ) -> None:
    #     """Update the tracker with a new detection and optional additional observation values.

    #     Args:
    #         detection (Union[Detection, None]): Detection object to update the tracker with, or None if no detection is available.
    #         global_step (Optional[int], optional): The global step counter for the tracking process. Defaults to None.
    #         **kwargs: Additional keyword arguments containing observation values to update, which will overwrite the values from the detection object if there are any overlaps.

    #     Note:
    #         If there is no detection (i.e., detection is None), the tracker will still update the observation with None values.
    #         Additional observation values provided through keyword arguments will still be updated even if detection is None.

    #     Example:
    #         # Assuming tracker is an instance of the SingleObjectTracker class and detection is a Detection object
    #         tracker.update(detection, global_step=5, velocity=0.8)
    #     """

    #     self.steps_alive += 1
    #     if global_step is not None:
    #         self.global_step = int(global_step)
    #     else:
    #         self.global_step += 1

    #     if detection is not None:
    #         self.steps_positive += 1
    #         self.staleness = 0.0
    #         self.update_observation(detection, **kwargs)
    #     else:
    #         self.staleness += 1
    #         self.update_observation(None, **kwargs)

    def print(self, num_recent_obs: int = 1, use_colors: bool = False) -> None:
        """
        Pretty-print the Tracklet information.

        Args:
            num_recent_obs (int, optional): The number of recent observations to display for each observation type. Defaults to 1.
            use_colors (bool, optional): Whether to use colors in the output. Defaults to False.
        """
        WHITE = ""
        if use_colors:
            ENDC = "\033[0m"
            id_color = id_to_color(str(self.id))
        else:
            ENDC = ""
            id_color = ""

        title = f"Tracklet(id={self.id}, steps_alive={self.steps_alive}, staleness={self.staleness}, is_active={self.is_active()})"
        max_name_length = max([len(name) for name in self._observations.keys()])
        max_values_length = max(
            [len(", ".join([str(val) for val in obs[-num_recent_obs:]])) for obs in self._observations.values()]
        )

        box_width = max(len(title) + 4, max_name_length + max_values_length + 7)
        box_width = min(box_width, 100)

        message = f"{id_color}{'╔' + '═' * box_width + '╗'}\n"
        message += f"║ {title}{' ' * (box_width - len(title) - 1)}║\n"
        message += f"{'╟' + '─' * box_width + '╢'}{ENDC}\n"
        for name, obs in self._observations.items():
            recent_values = obs[-num_recent_obs:] if obs else []
            values_str = ", ".join(
                [f"{WHITE}{str(val)[:60]}{ENDC}" if len(str(val)) > 60 else str(val) for val in recent_values]
            )
            message += f"{id_color}║ {ENDC}"
            message += f"{WHITE} {name}: [{values_str}]{' ' * (box_width - len(name) - len(values_str) - 6)}{ENDC}"
            message += f"{id_color}║{ENDC}\n"
        message += f"{id_color}{'╚' + '═' * box_width + '╝'}{ENDC}"
        print(message)

    @property
    def states(self) -> Dict[str, Any]:
        """Get the current state of the tracker.

        Returns:
            Dict[str, Any]: A dictionary containing the current state of the tracker.
        """
        return self._states

    # def _update_observation(self, detection: Union[Detection, None], **kwargs) -> None:
    #     """Update all registered observation types with values from a detection object or keyword arguments.

    #     This method updates the values of all registered observation types for the tracker. If a detection object is provided, its attributes will be used to update the corresponding observations. Additionally, any keyword arguments passed to this method can be used to update the observation values, taking precedence over the values from the detection object.

    #     Args:
    #         detection (Union[Detection, None]): A Detection object containing the new values for the registered observation types,or None if no detection is available.
    #         **kwargs: Additional keyword arguments containing observation values to update, which will overwrite the values from the detection object if there are any overlaps.

    #     Raises:
    #         KeyError: If an observation key in the kwargs does not match any registered observation types.

    #     Example:
    #         # Assuming tracker is an instance of the SingleObjectTracker class and detection is a Detection object
    #         tracker._update_observation(detection, velocity=0.8)
    #     """
    #     new_observations = (
    #         {
    #             "box": detection.box,
    #             "score": detection.score,
    #             "class_id": detection.class_id,
    #             "feature": detection.feature,
    #         }
    #         if detection is not None
    #         else {}
    #     )
    #     new_observations.update(kwargs)

    #     for key in self._observations:
    #         if key in new_observations:
    #             self.update_observation(key, new_observations[key])
    #         else:
    #             self.update_observation(key, None)
