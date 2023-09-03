from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, Union

import numpy as np
import optuna

from sportslabkit import Tracklet
from sportslabkit.dataframe.bboxdataframe import BBoxDataFrame
from sportslabkit.logger import logger
from sportslabkit.metrics.object_detection import iou_scores


class SingleObjectTracker(ABC):
    def __init__(self, target, window_size=1, step_size=None, pre_init_args={}, post_init_args={}):
        self.target = target
        self.init_target = target
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

    def update_tracklet_observations(self, states: Dict[str, Any]):
        self.check_required_types(states)
        for required_type in self.required_keys:
            self.tracklet.update_observation(required_type, states[required_type])
        self.tracklet.increment_counter()

    @abstractmethod
    def update(self, current_frame: Any) -> Dict[str, Any]:
        pass

    def process_sequence_item(self, sequence: Any):
        is_batched = isinstance(sequence, np.ndarray) and len(sequence.shape) == 4
        if is_batched:
            updated_states = self.update(sequence)
        else:
            updated_states = [self.update(sequence)]

        for updated_state in updated_states:
            self.check_updated_state(updated_state)
            self.update_tracklet_observations(updated_state)
            self.frame_count += 1

    def track(self, sequence: Union[Iterable[Any], np.ndarray]) -> Tracklet:
        if not isinstance(sequence, (Iterable, np.ndarray)):
            raise ValueError("Input 'sequence' must be an iterable or numpy array of frames/batches")

        self.pre_track()
        for i in range(0, len(sequence) - self.window_size + 1, self.step_size):
            logger.debug(f"Processing frames {i} to {i + self.window_size}")
            self.process_sequence_item(sequence[i : i + self.window_size])
        self.post_track()
        return self.tracklet

    def pre_track(self):
        # Hook that subclasses can override
        pass

    def post_track(self):
        pass

    def reset(self):
        self.pre_initialize(**self.pre_init_args)

        # Initialize the single object tracker
        logger.debug("Initializing tracker...")
        self.tracklet = Tracklet()
        for required_type in self.required_keys:
            self.tracklet.register_observation_type(required_type)
        self.frame_count = 0
        self.update_tracklet_observations(self.init_target)

        self.post_initialize(**self.post_init_args)
        logger.debug("Tracker initialized.")

    def check_required_types(self, target: Dict[str, Any]):
        missing_types = [required_type for required_type in self.required_keys if required_type not in target]

        if missing_types:
            required_types_str = ", ".join(self.required_keys)
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

        missing_types = [required_type for required_type in self.required_keys if required_type not in state]

        if missing_types:
            missing_types_str = ", ".join(missing_types)
            raise ValueError(
                f"The returned state from `update` is missing the following required types: {missing_types_str}."
            )

    @property
    def required_keys(self):
        raise NotImplementedError

    @property
    def hparam_searh_space(self):
        return {}

    def create_hparam_dict(self):
        # Create a dictionary for all hyperparameters
        hparams = {"self": self.hparam_search_space} if hasattr(self, "hparam_search_space") else {}
        for attribute in vars(self):
            value = getattr(self, attribute)
            if hasattr(value, "hparam_search_space") and attribute not in hparam_search_space:
                hparams[attribute] = {}
                search_space = value.hparam_search_space
                for param_name, param_space in search_space.items():
                    hparams[attribute][param_name] = {
                        "type": param_space["type"],
                        "values": param_space.get("values"),
                        "low": param_space.get("low"),
                        "high": param_space.get("high"),
                    }
        return hparams

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
                        params[attribute][param_name] = trial.suggest_categorical(param_name, param_values["values"])
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
                        raise ValueError(f"Unknown parameter type: {param_values['type']}")

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

        # check that the ground truth positions are in the correct format
        if isinstance(ground_truth_positions, BBoxDataFrame):
            ground_truth_positions = np.expand_dims(ground_truth_positions.values, axis=1)[:, :, :4]

        hparams = self.create_hparam_dict()

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
