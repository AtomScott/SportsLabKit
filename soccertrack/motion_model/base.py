from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, Union

import numpy as np

from soccertrack import Tracklet
from soccertrack.logger import logger


class MotionModel(ABC):
    """Abstract base class for motion models.

    This class defines a common interface for all motion models.
    Derived classes should implement the update, and predict methods. MotionModels are procedural and stateless. The state of tracklet is managed by the Tracklet class. The tracklet must have the required observations and states for the motion model to work. If the tracklet doesn't have the required observations or states, the motion model will raise an error and tell the user which observations or states are missing.
    """

    hparam_search_space: Dict[str, Type] = {}
    required_observation_types: List[str] = NotImplemented
    required_state_types: List[str] = NotImplemented

    def __init__(self):
        """Initialize the MotionModel."""
        self.name = self.__class__.__name__

    def __call__(self, tracklet: Tracklet) -> Any:
        """Call the motion model to update its state and return the prediction.

        Args:
            tracklet (Tracklet): The single object tracker instance.

        Returns:
            Any: The predicted state after updating the motion model.
        """
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

    def update(self, observations: Dict[str, Any], states: Dict[str, Any]) -> None:
        """Update the motion model's internal state.

        Args:
            observations (Dict[str, Any]): The observations to update the motion model with.
            states (Dict[str, Any]): The states to update the motion model with.
        """

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
    def from_config(cls: Type["MotionModel"], config: Dict) -> "MotionModel":
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
                raise KeyError(
                    f"{self.name} requires observation type `{obs_type}` but it is not registered."
                )
            if len(tracklet._observations[obs_type]) == 0:
                raise KeyError(
                    f"{self.name} requires observation type `{obs_type}` but it is empty."
                )

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


# class BaseMotionModule(pl.LightningModule):
#     def __init__(self, model, learning_rate=1e-3, roll_out_steps=10, single_agent=False):
#         super().__init__()
#         self.model = model

#         self.learning_rate = learning_rate
#         self.roll_out_steps = roll_out_steps
#         self.single_agent = single_agent

#     def forward(self, x):
#         return self.model(x)

#     def roll_out(self, x, n_steps=None, y_gt=None):
#         n_steps = n_steps or self.roll_out_steps
#         return self.model.roll_out(x, n_steps, y_gt=y_gt)

#     def compute_loss(self, y_gt, y_pred):
#         mse = mean_squared_error(y_pred, y_gt)
#         rmse = torch.sqrt(mse)
#         return rmse

#     def compute_eval_metrics(self, y_gt, y_pred):
#         # calculate the mean absolute error at first step, 10%, 50% and 100% of the trajectory
#         total_steps = self.roll_out_steps
#         eval_steps = [
#             0,
#             int(0.1 * total_steps),
#             int(0.5 * total_steps),
#             total_steps - 1,
#         ]
#         eval_step_names = ["first_step", "10pct", "50pct", "100pct"]
#         eval_metrics = {}
#         for step, name in zip(eval_steps, eval_step_names):
#             rmse = torch.sqrt(mean_squared_error(y_pred[:, step], y_gt[:, step]))
#             eval_metrics[f"rmse_{name}"] = rmse
#         return eval_metrics

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         # use ground truth to roll out (teacher forcing)
#         y_pred = self.roll_out(x, y_gt=y)
#         loss = self.compute_loss(y_gt=y, y_pred=y_pred)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_pred = self.roll_out(x)
#         loss = self.compute_loss(y_gt=y, y_pred=y_pred)
#         self.log("val_loss", loss, prog_bar=True)
#         if batch_idx == 0:
#             x = x.detach().cpu().numpy()
#             y = y.detach().cpu().numpy()
#             y_pred = y_pred.detach().cpu().numpy()

#             # plot a trajectory from the first batch
#             self.plot_trajectory(x[0], y[0], y_pred[0])

#             # plot a histogram of the displacements
#             y_displacements = y[:, 1:] - y[:, :-1]
#             y_pred_displacements = y_pred[:, 1:] - y_pred[:, :-1]

#             y_displacements_x = y_displacements[:, :, 0].flatten()
#             y_displacements_y = y_displacements[:, :, 1].flatten()

#             y_pred_displacements_x = y_pred_displacements[:, :, 0].flatten()
#             y_pred_displacements_y = y_pred_displacements[:, :, 1].flatten()
#             self.plot_histogram(y_displacements_x, y_pred_displacements_x, "x")
#             self.plot_histogram(y_displacements_y, y_pred_displacements_y, "y")

#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         y_pred = self.roll_out(x)
#         eval_metrics = self.compute_eval_metrics(y_gt=y, y_pred=y_pred)

#         for name, value in eval_metrics.items():
#             self.log(name, value, prog_bar=False)

#     def plot_trajectory(self, x, y, y_pred):
#         current_epoch = self.current_epoch
#         title = f"Trajectory at epoch {current_epoch}"
#         plt.figure(figsize=(10, 10))

#         if self.single_agent:
#             plt.scatter(x[:, 0], x[:, 1], label="input")
#             plt.scatter(y[:, 0], y[:, 1], label="ground truth")
#             plt.scatter(y_pred[:, 0], y_pred[:, 1], label="prediction")
#         else:
#             plt.scatter(x[:, :, 0], x[:, :, 1], label="input")
#             plt.scatter(y[:, :, 0], y[:, :, 1], label="ground truth")
#             plt.scatter(y_pred[:, :, 0], y_pred[:, :, 1], label="prediction")
#         plt.title(title)
#         plt.legend()
#         self.logger.experiment.add_figure("trajectory", plt.gcf(), current_epoch)
#         plt.close("all")

#     def plot_histogram(self, y_displacements, y_pred_displacements, tag):
#         current_epoch = self.current_epoch

#         bins = np.linspace(-1, 1, 300)
#         title = f"{tag} displacements at epoch {current_epoch}"
#         plt.figure(figsize=(10, 10))
#         plt.hist(y_displacements, label="ground truth", bins=bins)
#         plt.hist(y_pred_displacements, label="prediction", bins=bins)
#         plt.title(title)
#         plt.legend()
#         self.logger.experiment.add_figure("displacements", plt.gcf(), current_epoch)
#         plt.close("all")

#     def configure_optimizers(self):
#         optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, mode="min", factor=0.3, patience=10, min_lr=1e-6
#         )
#         monitor = "val_loss"  # Specify the metric to monitor
#         return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": monitor}
