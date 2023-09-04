from __future__ import annotations

import json
from ast import literal_eval
from typing import Any, Iterable, Mapping, Optional

import cv2
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation
from mplsoccer import Pitch
from numpy.typing import ArrayLike, NDArray

from sportslabkit.dataframe.base import BaseSLKDataFrame
from sportslabkit.logger import logger
from sportslabkit.types.types import _pathlike


def merge_dicts(*dicts):
    """Merge dictionaries.

    Later dictionaries take precedence.
    """
    merged = {}
    for d in dicts:
        if d is not None:
            merged.update(d)
    return merged


class CoordinatesDataFrame(BaseSLKDataFrame, pd.DataFrame):
    _metadata = [
        "source_keypoints",
        "target_keypoints",
    ]

    @property
    def _constructor(self):
        return CoordinatesDataFrame

    @property
    def H(self) -> NDArray[np.float64]:
        """Calculate the homography transformation matrix from pitch to video
        space.

        Returns:
            NDArray[np.float64]: homography transformation matrix.
        """
        H, *_ = cv2.findHomography(self.source_keypoints, self.target_keypoints, cv2.RANSAC, 5.0)
        return H

    def set_keypoints(
        self,
        source_keypoints: Optional[ArrayLike] = None,
        target_keypoints: Optional[ArrayLike] = None,
        mapping: Optional[Mapping] = None,
        mapping_file: Optional[_pathlike] = None,
    ) -> None:
        """Set the keypoints for the homography transformation. Make sure that
        the target keypoints are the pitch coordinates. Also each keypoint must
        be a tuple of (Lon, Lat) or (x, y) coordinates.

        Args:
            source_keypoints (Optional[ArrayLike], optional): Keypoints in pitch space. Defaults to None.
            target_keypoints (Optional[ArrayLike], optional): Keypoints in video space. Defaults to None.
        """

        if mapping_file is not None:
            with open(mapping_file, "r") as f:
                mapping = json.load(f)
        if mapping is not None:
            target_keypoints, source_keypoints = [], []
            for target_kp, source_kp in mapping.items():
                if isinstance(target_kp, str):
                    target_kp = literal_eval(target_kp)
                if isinstance(source_kp, str):
                    source_kp = literal_eval(source_kp)
                target_keypoints.append(target_kp)
                source_keypoints.append(source_kp)

        self.source_keypoints = np.array(source_keypoints)
        self.target_keypoints = np.array(target_keypoints)

    def to_pitch_coordinates(self, drop=True):
        """Convert image coordinates to pitch coordinates."""
        transformed_groups = []
        for i, g in self.iter_players():
            pts = g[[(i[0], i[1], "Lon"), (i[0], i[1], "Lat")]].values
            x, y = cv2.perspectiveTransform(np.asarray([pts]), self.H).squeeze().T
            g[(i[0], i[1], "x")] = x
            g[(i[0], i[1], "y")] = y

            if drop:
                g.drop(columns=[(i[0], i[1], "Lon"), (i[0], i[1], "Lat")], inplace=True)
            transformed_groups.append(g)

        return self._constructor(pd.concat(transformed_groups, axis=1))

    # def visualize_frames

    @staticmethod
    def from_numpy(
        arr: np.ndarray,
        team_ids: Optional[Iterable[str]] = None,
        player_ids: Optional[Iterable[int]] = None,
        attributes: Optional[Iterable[str]] = ("x", "y"),
        auto_fix_columns: bool = True,
    ):
        """Create a CoordinatesDataFrame from a numpy array of either shape (L, N, 2) or (L, N * 2) where L is the number of frames, N is the number of players and 2 is the number of coordinates (x, y).

        Args:
            arr : Numpy array.
            team_ids : Team ids. Defaults to None. If None, team ids will be set to 0 for all players. If not None, must have the same length as player_ids
            Player ids: Player ids. Defaults to None. If None, player ids will be set to 0 for all players. If not None, must have the same length as team_ids
            attributes : Attribute names to use. Defaults to ("x", "y").
            auto_fix_columns : If True, will automatically fix the team_ids, player_ids and attributes so that they are equal to the number of columns. Defaults to True.


        Returns:
            CoordinatesDataFrame: CoordinatesDataFrame.

        Examples:
            >>> from soccertrack.dataframe import CoordinatesDataFrame
            >>> import numpy as np
            >>> arr = np.random.rand(10, 22, 2)
            >>> codf = CoordinatesDataFrame.from_numpy(arr, team_ids=["0"] * 22, player_ids=list(range(22)))

        """
        n_frames, n_players, n_attributes = arr.shape
        n_columns = n_players * n_attributes

        if team_ids and player_ids:
            assert len(team_ids) == len(
                player_ids
            ), f"team_ids and player_ids must have the same length. Got {len(team_ids)} and {len(player_ids)} respectively."

        assert arr.ndim in (2, 3), "Array must be of shape (L, N, 2) or (L, N * 2)"
        if arr.ndim == 3:
            arr = arr.reshape(arr.shape[0], -1)

        df = pd.DataFrame(arr)

        if team_ids is None:
            if n_players == 23:
                team_ids = ["0"] * 22 + ["1"] * 22 + ["ball"] * 2
            else:
                team_ids = ["0"] * n_players * n_attributes
        elif auto_fix_columns and len(team_ids) != n_columns:
            team_ids = np.repeat(team_ids, n_attributes)

        if player_ids is None:
            if n_players == 23:
                _players = list(np.linspace(0, 10, 22).round().astype(int))
                player_ids = _players + _players + [0, 0]
            else:
                player_ids = list(range(n_players)) * n_attributes
        elif auto_fix_columns and len(player_ids) != player_ids:
            player_ids = np.repeat(player_ids, n_attributes)

        attributes = attributes * n_players

        def _assert_correct_length(x, key):
            assert (
                len(x) == n_columns
            ), f"Incorrect number of resulting {key} columns: {len(x)} != {n_columns}. Set auto_fix_columns to False to disable automatic fixing of columns. See docs for more information."

        _assert_correct_length(team_ids, "TeamID")
        _assert_correct_length(player_ids, "PlayerID")
        _assert_correct_length(attributes, "Attributes")

        idx = pd.MultiIndex.from_arrays(
            [team_ids, player_ids, attributes],
        )

        # change multicolumn
        df = CoordinatesDataFrame(df.values, index=df.index, columns=idx)

        df.rename_axis(["TeamID", "PlayerID", "Attributes"], axis=1, inplace=True)
        df.index.name = "frame"

        return CoordinatesDataFrame(df)

    @staticmethod
    def from_dict(d: dict, attributes: Optional[Iterable[str]] = ("x", "y")):
        """Create a CoordinatesDataFrame from a nested dictionary contating the coordinates of the players and the ball.

        The input dictionary should be of the form:
        {
            home_team_key: {
                PlayerID: {frame: [x, y], ...},
                PlayerID: {frame: [x, y], ...},
                ...
            },
            away_team_key: {
                PlayerID: {frame: [x, y], ...},
                PlayerID: {frame: [x, y], ...},
                ...
            },
            ball_key: {
                frame: [x, y],
                frame: [x, y],
                ...
            }
        }
        The `PlayerID` can be any unique identifier for the player, e.g. their jersey number or name. The PlayerID for the ball can be omitted, as it will be set to "0". `frame` must be an integer identifier for the frame number.

        Args:
            dict (dict): Nested dictionary containing the coordinates of the players and the ball.
            attributes (Optional[Iterable[str]], optional): Attributes to use for the coordinates. Defaults to ("x", "y").

        Returns:
            CoordinatesDataFrame: CoordinatesDataFrame.
        """
        attributes = list(attributes)
        data = []
        for team, team_dict in d.items():
            for player, player_dict in team_dict.items():
                for frame, coords in player_dict.items():
                    data.append([team, player, frame, *coords])

        df = pd.DataFrame(
            data,
            columns=["TeamID", "PlayerID", "frame", *attributes],
        )

        df = df.pivot(index="frame", columns=["TeamID", "PlayerID"], values=attributes)
        multi_index = pd.MultiIndex.from_tuples(df.columns.swaplevel(0, 1).swaplevel(1, 2))
        df.columns = pd.MultiIndex.from_tuples(multi_index)
        df.rename_axis(["TeamID", "PlayerID", "Attributes"], axis=1, inplace=True)
        df.sort_index(axis=1, inplace=True)

        return CoordinatesDataFrame(df)

    def visualize_frame(
        self,
        frame_idx: int,
        save_path: Optional[_pathlike] = None,
        ball_key: str = "ball",
        home_key: str = "0",
        away_key: str = "1",
        marker_kwargs: Optional[dict[str, Any]] = None,
        ball_kwargs: Optional[dict[str, Any]] = None,
        home_kwargs: Optional[dict[str, Any]] = None,
        away_kwargs: Optional[dict[str, Any]] = None,
        save_kwargs: Optional[dict[str, Any]] = None,
    ):
        """Visualize a single frame.

        Visualize a frame given a frame number and save it to a path. The `CoordinatesDataFrame` is expected to already have been normalized so that the pitch is 105x68, e.g. coordinates on the x-axis range from 0 to 105 and coordinates on the y-axis range from 0 to 68.

        Similarly, you can pass keyword arguments to change the appearance of the markers. For example, to change the size of the markers, you can pass `ms=6` to `away_kwargs` by, e.g. `codf.visualize_frames("animation.gif", away_kwargs={"ms": 6})`. See the `matplotlib.pyplot.plot` documentation for more information. Note that `marker_kwargs` will be used for all markers but will be overwritten by `ball_kwargs`, `home_kwargs` and `away_kwargs` if a dictionary with the same key is passed (later dictionaries take precedence).

        Args:
            frame_idx: Frame number.
            save_path: Path to save the image. Defaults to None.
            ball_key: Key (TeamID) for the ball. Defaults to "ball".
            home_key: Key (TeamID) for the home team. Defaults to "0".
            away_key: Key (TeamID) for the away team. Defaults to "1".
            marker_kwargs: Keyword arguments for the markers.
            ball_kwargs: Keyword arguments specifically for the ball marker.
            home_kwargs: Keyword arguments specifically for the home team markers.
            away_kwargs: Keyword arguments specifically for the away team markers.
            save_kwargs: Keyword arguments for the save function.

        Note:
            `marker_kwargs` will be used for all markers but will be overwritten by `ball_kwargs`, `home_kwargs` and `away_kwargs`. All keyword arguments are passed to `plt.plot`. `save_kwargs` are passed to `plt.savefig`.

        Warning:
            All keyword arguments are passed to `plt.plot`. If you pass an invalid keyword argument, you will get an error.

        Example:
            >>> codf = CoordinatesDataFrame.from_numpy(np.random.randint(0, 105, (1, 23, 2)))
            >>> codf.visualize_frame(0)

        .. image:: /_static/visualize_frame.png
        """

        _marker_kwargs = merge_dicts(
            {"marker": "o", "markeredgecolor": "None", "linestyle": "None"},
            marker_kwargs,
        )

        _ball_kwargs = merge_dicts(
            _marker_kwargs,
            {"zorder": 3, "ms": 6, "markerfacecolor": "w"},
            marker_kwargs,
            ball_kwargs,
        )
        _home_kwargs = merge_dicts(
            _marker_kwargs,
            {"zorder": 10, "ms": 10, "markerfacecolor": "b"},
            marker_kwargs,
            home_kwargs,
        )

        _away_kwargs = merge_dicts(
            _marker_kwargs,
            {"zorder": 10, "ms": 10, "markerfacecolor": "r"},
            marker_kwargs,
            away_kwargs,
        )

        _save_kwargs = merge_dicts({"facecolor": "black", "pad_inches": 0.0}, save_kwargs)

        _df = self.copy()
        _df = _df[_df.index == frame_idx]

        df_ball = _df[ball_key]
        df_home = _df[home_key]
        df_away = _df[away_key]
        pitch = Pitch(
            pitch_color="black",
            line_color=(0.3, 0.3, 0.3),
            pitch_type="custom",
            pitch_length=105,
            pitch_width=68,
            label=False,
        )

        fig, ax = pitch.draw(figsize=(8, 5.2))

        ax.plot(
            df_ball.loc[:, (slice(None), "x")],
            df_ball.loc[:, (slice(None), "y")],
            **_ball_kwargs,
        )
        ax.plot(
            df_away.loc[:, (slice(None), "x")],
            df_away.loc[:, (slice(None), "y")],
            **_away_kwargs,
        )
        ax.plot(
            df_home.loc[:, (slice(None), "x")],
            df_home.loc[:, (slice(None), "y")],
            **_home_kwargs,
        )

        if save_path is not None:
            fig.savefig(save_path, **_save_kwargs)

    def visualize_frames(
        self,
        save_path: _pathlike,
        ball_key: str = "ball",
        home_key: str = "0",
        away_key: str = "1",
        marker_kwargs: Optional[dict[str, Any]] = None,
        ball_kwargs: Optional[dict[str, Any]] = None,
        home_kwargs: Optional[dict[str, Any]] = None,
        away_kwargs: Optional[dict[str, Any]] = None,
        save_kwargs: Optional[dict[str, Any]] = None,
    ):
        """Visualize multiple frames using matplotlib.animation.FuncAnimation.

        Visualizes the frames and generates a pitch animation. The `CoordinatesDataFrame` is expected to already have been normalized so that the pitch is 105x68, e.g. coordinates on the x-axis range from 0 to 105 and coordinates on the y-axis range from 0 to 68.

        To customize the animation, you can pass keyword arguments to `matplotlib.animation.FuncAnimation`. For example, to change the frame rate, you can pass `fps=30` to `save_kwargs` by, e.g. `codf.visualize_frames("animation.gif", save_kwargs={"fps": 30})`. See the `matplotlib.animation.FuncAnimation` documentation for more information.

        Similarly, you can pass keyword arguments to change the appearance of the markers. For example, to change the size of the markers, you can pass `ms=6` to `away_kwargs` by, e.g. `codf.visualize_frames("animation.gif", away_kwargs={"ms": 6})`. See the `matplotlib.pyplot.plot` documentation for more information.  Note that `marker_kwargs` will be used for all markers but will be overwritten by `ball_kwargs`, `home_kwargs` and `away_kwargs` if a dictionary with the same key is passed (later dictionaries take precedence).

        Args:
            frame_idx: Frame number.
            save_path: Path to save the image. Defaults to None.
            ball_key: Key (TeamID) for the ball. Defaults to "ball".
            home_key: Key (TeamID) for the home team. Defaults to "0".
            away_key: Key (TeamID) for the away team. Defaults to "1".
            marker_kwargs: Keyword arguments for the markers.
            ball_kwargs: Keyword arguments specifically for the ball marker.
            home_kwargs: Keyword arguments specifically for the home team markers.
            away_kwargs: Keyword arguments specifically for the away team markers.
            save_kwargs: Keyword arguments for the save function.

        Note:
            `marker_kwargs` will be used for all markers but will be overwritten by `ball_kwargs`, `home_kwargs` and `away_kwargs`. All keyword arguments are passed to `plt.plot`. `save_kwargs` are passed to `FuncAnimation.save`.

        Warning:
            All keyword arguments are passed either to `plt.plot` and `FuncAnimation.save`. If you pass an invalid keyword argument, you will get an error.

        Example:
            >>> codf = load_codf("/path/to/codf.csv")
            >>> codf.visualize_frames("/path/to/save.mp4")
            ...
            # Heres a demo using random data
            >>> codf = CoordinatesDataFrame.from_numpy(np.random.randint(0, 50, (1, 23, 2)))
            >>> codf = codf.loc[codf.index.repeat(5)] # repeat the same frame 5 times
            >>> codf += np.array([[0,1,2,3,4]]).T # add some movment
            >>> codf.visualize_frames('visualize_frames.gif', save_kwargs={'fps':2})

        .. image:: /_static/visualize_frames.gif
        """
        _marker_kwargs = merge_dicts(
            {"marker": "o", "markeredgecolor": "None", "linestyle": "None"},
            marker_kwargs,
        )

        _ball_kwargs = merge_dicts(
            _marker_kwargs,
            {"zorder": 3, "ms": 6, "markerfacecolor": "w"},
            marker_kwargs,
            ball_kwargs,
        )
        _home_kwargs = merge_dicts(
            _marker_kwargs,
            {"zorder": 10, "ms": 10, "markerfacecolor": "b"},
            marker_kwargs,
            home_kwargs,
        )

        _away_kwargs = merge_dicts(
            _marker_kwargs,
            {"zorder": 10, "ms": 10, "markerfacecolor": "r"},
            marker_kwargs,
            away_kwargs,
        )

        _save_kwargs = merge_dicts(
            {
                "dpi": 100,
                "fps": 10,
                "savefig_kwargs": {"facecolor": "black", "pad_inches": 0.0},
            },
            save_kwargs,
        )

        _df = self.copy()

        df_ball = _df[ball_key]
        df_home = _df[home_key]
        df_away = _df[away_key]
        pitch = Pitch(
            pitch_color="black",
            line_color=(0.3, 0.3, 0.3),
            pitch_type="custom",
            pitch_length=105,
            pitch_width=68,
            label=False,
        )

        fig, ax = pitch.draw(figsize=(8, 5.2))

        ball, *_ = ax.plot([], [], **_ball_kwargs)
        away, *_ = ax.plot([], [], **_away_kwargs)
        home, *_ = ax.plot([], [], **_home_kwargs)

        def animate(i):
            """Function to animate the data.

            Each frame it sets the data for the players and the ball.
            """
            # set the ball data with the x and y positions for the ith frame
            ball.set_data(
                df_ball.loc[:, (slice(None), "x")].iloc[i],
                df_ball.loc[:, (slice(None), "y")].iloc[i],
            )

            # set the player data using the frame id
            away.set_data(
                df_away.loc[:, (slice(None), "x")].iloc[i],
                df_away.loc[:, (slice(None), "y")].iloc[i],
            )
            home.set_data(
                df_home.loc[:, (slice(None), "x")].iloc[i],
                df_home.loc[:, (slice(None), "y")].iloc[i],
            )
            return ball, away, home

        anim = FuncAnimation(fig, animate, frames=len(_df), blit=True)

        try:
            anim.save(save_path, **_save_kwargs)
        except Exception:
            logger.error(
                "BrokenPipeError: Saving animation failed, which might be an ffmpeg problem. Trying again with different codec."
            )
            _save_kwargs["extra_args"] = ["-vcodec", "mpeg4", "-pix_fmt", "yuv420p"]
            try:
                anim.save(save_path, **_save_kwargs)
            except Exception as e:
                logger.error("Saving animation failed again. Exiting without saving the animation.")
                print(e)

    # @property
    # def _constructor_sliced(self):
    #     raise NotImplementedError("This pandas method constructs pandas.Series object, which is not yet implemented in {self.__name__}.")
