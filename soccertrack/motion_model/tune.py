import optuna
import numpy as np
from soccertrack import Tracklet
from soccertrack.metrics import iou_score, convert_to_x1y1x2y2


def tune_motion_model(
    motion_model_class,
    detections,
    ground_truth_positions,
    n_trials=100,
    hparam_search_space=None,
    metric=iou_score,
    verbose=False,
    return_study=False,
):
    def objective(trial: optuna.Trial):
        params = {}
        for param_name, search_space in hparam_search_space.items():
            if search_space["type"] == "categorical":
                params[param_name] = trial.suggest_categorical(param_name, search_space["values"])
            elif search_space["type"] == "float":
                params[param_name] = trial.suggest_float(param_name, search_space["low"], search_space["high"])
            elif search_space["type"] == "logfloat":
                params[param_name] = trial.suggest_float(
                    param_name, search_space["low"], search_space["high"], log=True
                )
            elif search_space["type"] == "int":
                params[param_name] = trial.suggest_int(param_name, search_space["low"], search_space["high"])

        motion_model = motion_model_class(**params)
        tracklet = Tracklet()
        tracklet.register_observation_types(["box", "score"])
        ious = []

        for det, gt in zip(detections, ground_truth_positions):
            obs = {"box": det.box, "score": det.score}
            tracklet.update_observations(obs)
            prediction = motion_model(tracklet)

            iou = iou_score(convert_to_x1y1x2y2(prediction), convert_to_x1y1x2y2(gt))
            ious.append(iou)

        avg_iou = np.mean(ious)
        return 1 - avg_iou  # Minimize 1 - IoU

    if verbose:
        optuna.logging.set_verbosity(optuna.logging.INFO)
    else:
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    if hparam_search_space is None:
        hparam_search_space = motion_model_class.hparam_search_space

    study = optuna.create_study()
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_iou = 1 - study.best_value
    if return_study:
        return best_params, best_iou, study
    return best_params, best_iou
