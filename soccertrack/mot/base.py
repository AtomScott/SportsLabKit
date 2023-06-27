from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np
import optuna
import pandas as pd

from soccertrack import Tracklet
from soccertrack.dataframe.bboxdataframe import BBoxDataFrame
from soccertrack.logger import logger
from soccertrack.metrics.object_detection import iou_scores


class MultiObjectTracker(ABC):
    def __init__(
        self, window_size=1, step_size=None, pre_init_args={}, post_init_args={}
    ):
        self.window_size = window_size
        self.step_size = step_size or window_size

        self.pre_init_args = pre_init_args
        self.post_init_args = post_init_args
        self.reset()

    def pre_initialize(self, **kwargs):
        # Hook that subclasses can override
        pass

    def post_initialize(self, **kwargs):
        # Hook that subclasses can override
        pass

    def update_tracklet(self, tracklet: Tracklet, states: Dict[str, Any]):
        self._check_required_observations(states)
        tracklet.update_observations(states, self.frame_count)
        tracklet.increment_counter()
        return tracklet

    @abstractmethod
    def update(
        self, current_frame: Any, trackelts: List[Tracklet]
    ) -> Tuple[List[Tracklet], List[Dict[str, Any]]]:
        pass

    def process_sequence_item(self, sequence: Any):
        self.frame_count += 1  # incremenmt first to match steps alive
        is_batched = isinstance(sequence, np.ndarray) and len(sequence.shape) == 4
        tracklets = self.tracklets
        if is_batched:
            raise NotImplementedError("Batched tracking is not yet supported")

        assigned_tracklets, new_tracklets, unassigned_tracklets = self.update(
            sequence, tracklets
        )
        logger.debug(
            f"assigned_tracklets: {len(assigned_tracklets)}, new_tracklets: {len(new_tracklets)}, unassigned_tracklets: {len(unassigned_tracklets)}"
        )
        self.tracklets = assigned_tracklets + new_tracklets

        # Update the dead tracklets with nan values
        self.dead_tracklets += unassigned_tracklets

    def track(self, sequence: Union[Iterable[Any], np.ndarray]) -> Tracklet:
        if not isinstance(sequence, (Iterable, np.ndarray)):
            raise ValueError(
                "Input 'sequence' must be an iterable or numpy array of frames/batches"
            )

        self.pre_track()
        for i in range(0, len(sequence) - self.window_size + 1, self.step_size):
            self.process_sequence_item(sequence[i : i + self.window_size].squeeze())
        self.post_track()
        return self.tracklets

    def pre_track(self):
        # Hook that subclasses can override
        pass

    def post_track(self):
        pass

    def reset(self):
        self.pre_initialize(**self.pre_init_args)

        # Initialize the single object tracker
        logger.debug("Initializing tracker...")
        self.tracklets = []
        self.dead_tracklets = []
        self.frame_count = 0

        self.post_initialize(**self.post_init_args)
        logger.debug("Tracker initialized.")

    def _check_required_observations(self, target: Dict[str, Any]):
        missing_types = [
            required_type
            for required_type in self.required_observation_types
            if required_type not in target
        ]

        if missing_types:
            required_types_str = ", ".join(self.required_observation_types)
            missing_types_str = ", ".join(missing_types)
            current_types_str = ", ".join(target.keys())

            raise ValueError(
                f"Input 'target' is missing the following required types: {missing_types_str}.\n"
                f"Required types: {required_types_str}\n"
                f"Current types in 'target': {current_types_str}"
            )

    def check_updated_state(self, state: Dict[str, Any]):
        if not isinstance(state, dict):
            raise ValueError("The `update` method must return a dictionary.")

        missing_types = [
            required_type
            for required_type in self.required_observation_types
            if required_type not in state
        ]

        if missing_types:
            missing_types_str = ", ".join(missing_types)
            raise ValueError(
                f"The returned state from `update` is missing the following required types: {missing_types_str}."
            )

    def create_tracklet(self, state: Dict[str, Any]):
        tracklet = Tracklet()
        for required_type in self.required_observation_types:
            tracklet.register_observation_type(required_type)

        self._check_required_observations(state)
        self.update_tracklet(tracklet, state)
        return tracklet

    def to_bbdf(self):
        """Create a bounding box dataframe."""
        df = pd.concat([t.to_bbdf() for t in self.tracklets], axis=1).sort_index()
        df = df.reindex(index=range(self.frame_count))

        all_tracklets = self.tracklets + self.dead_tracklets
        return pd.concat([t.to_bbdf() for t in all_tracklets], axis=1).sort_index()

    @property
    def required_observation_types(self):
        raise NotImplementedError

    @property
    def required_state_types(self):
        raise NotImplementedError

    @property
    def hparam_searh_space(self):
        return {}

    def tune_hparams(
        self,
        frames,
        ground_truth_positions,
        n_trials=100,
        hparam_search_space=None,
        metric=iou_scores,
        verbose=False,
        return_study=False,
    ):
        def objective(trial: optuna.Trial):
            params = {}
            for attribute, param_space in hparams.items():
                params[attribute] = {}
                for param_name, param_values in param_space.items():
                    if param_values["type"] == "categorical":
                        params[attribute][param_name] = trial.suggest_categorical(
                            param_name, param_values["values"]
                        )
                    elif param_values["type"] == "float":
                        params[attribute][param_name] = trial.suggest_float(
                            param_name, param_values["low"], param_values["high"]
                        )
                    elif param_values["type"] == "logfloat":
                        params[attribute][param_name] = trial.suggest_float(
                            param_name,
                            param_values["low"],
                            param_values["high"],
                            log=True,
                        )
                    elif param_values["type"] == "int":
                        params[attribute][param_name] = trial.suggest_int(
                            param_name, param_values["low"], param_values["high"]
                        )
                    else:
                        raise ValueError(
                            f"Unknown parameter type: {param_values['type']}"
                        )

            # Apply the hyperparameters to the attributes of `self`
            for attribute, param_values in params.items():
                for param_name, param_value in param_values.items():
                    if attribute == "self":
                        setattr(self, param_name, param_value)
                    else:
                        setattr(getattr(self, attribute), param_name, param_value)

            self.reset()
            tracklet = self.track(frames)
            predictions = tracklet.get_observations("box")

            # Fixme: Should not allow multiple ground truth targets for single object tracking
            ground_truth_targets = [gt[0] for gt in ground_truth_positions]

            score = iou_scores(predictions, ground_truth_targets, xywh=True)
            return score

        if hparam_search_space is None:
            hparam_search_space = {}

        # check that the ground truth positions are in the correct format
        print(ground_truth_positions.shape)
        if isinstance(ground_truth_positions, BBoxDataFrame):
            ground_truth_positions = np.expand_dims(
                ground_truth_positions.values, axis=1
            )[:, :, :4]

        # Create a dictionary for all hyperparameters
        hparams = (
            {"self": self.hparam_search_space}
            if hasattr(self, "hparam_search_space")
            else {}
        )
        for attribute in vars(self):
            value = getattr(self, attribute)
            if (
                hasattr(value, "hparam_search_space")
                and attribute not in hparam_search_space
            ):
                hparams[attribute] = {}
                search_space = value.hparam_search_space
                for param_name, param_space in search_space.items():
                    hparams[attribute][param_name] = {
                        "type": param_space["type"],
                        "values": param_space.get("values"),
                        "low": param_space.get("low"),
                        "high": param_space.get("high"),
                    }

        print("Hyperparameter search space: ")
        for attribute, param_space in hparams.items():
            print(f"{attribute}:")
            for param_name, param_values in param_space.items():
                print(f"\t{param_name}: {param_values}")
        if verbose:
            optuna.logging.set_verbosity(optuna.logging.INFO)
        else:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_iou = study.best_value
        if return_study:
            return best_params, best_iou, study
        return best_params, best_iou
