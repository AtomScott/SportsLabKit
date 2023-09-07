from abc import ABC, abstractmethod
from typing import Dict, Optional

import numpy as np
import pandas as pd

from sportslabkit.types import Color, Rect
from sportslabkit.viz.visualization import (
    draw_ellipse_with_caption,
    draw_marker,
    draw_rect,
    draw_rect_with_caption,
    get_color,
)


class BaseVisualizer(ABC):
    """Base class for visualizers."""

    @abstractmethod
    def draw_frame(self, frame_df: pd.DataFrame, frame: np.ndarray) -> np.ndarray:
        """Abstract method for drawing bounding boxes on a frame."""
        pass

class SimpleVisualizer(BaseVisualizer):
    """Simple class for visualizing bounding boxes on frames."""

    default_ball_style = {"color": (255, 255, 255), "thickness": 2}
    default_home_style = {"color": (0, 0, 255), "thickness": 2}
    default_away_style = {"color": (255, 0, 0), "thickness": 2}
    default_save_settings = {"dpi": 100, "fps": 10}

    def __init__(
        self,
        ball_key: str = "ball",
        home_key: str = "0",
        away_key: str = "1",
        ball_style: Optional[Dict] = None,
        home_style: Optional[Dict] = None,
        away_style: Optional[Dict] = None,
        save_settings: Optional[Dict] = None,
    ) -> None:
        self.ball_key = ball_key
        self.home_key = home_key
        self.away_key = away_key
        self.ball_style = {**self.default_ball_style, **(ball_style or {})}
        self.home_style = {**self.default_home_style, **(home_style or {})}
        self.away_style = {**self.default_away_style, **(away_style or {})}
        self.save_settings = {**self.default_save_settings, **(save_settings or {})}

    def _draw_object(self, frame: np.ndarray, box: Rect, style: Dict, caption: str) -> None:
        """Draw a single object on the frame."""
        color = get_color(style["color"])
        draw_rect(frame, box, color=color, thickness=style["thickness"])
        draw_rect_with_caption(frame, box, color=color, caption=caption, thickness=style["thickness"])

    def draw_frame(self, frame_df: pd.DataFrame, frame: np.ndarray) -> np.ndarray:
        """Draw bounding boxes from a DataFrame slice onto a frame."""
        frame = frame.copy()
        for (team_id, player_id), player_df in frame_df.iter_players():  # Assuming `iter_players` is a method in frame_df
            if player_df.isnull().any(axis=None):
                continue

            conf = player_df['conf'].values[0]
            caption = f"{team_id}:{player_id} {conf:.2f}%"

            box = Rect(
                int(player_df['bb_left']),
                int(player_df['bb_top']),
                int(player_df['bb_width']),
                int(player_df['bb_height'])
            )

            if team_id == self.home_key:
                self._draw_object(frame, box, self.home_style, caption)
            elif team_id == self.away_key:
                self._draw_object(frame, box, self.away_style, caption)
            elif team_id == self.ball_key:
                caption = f"BALL {conf:.2f}%"
                self._draw_object(frame, box, self.ball_style, caption)

        return frame


class FancyVisualizer(BaseVisualizer):
    """Fancy class for visualizing bounding boxes on frames with additional features."""

    default_ball_style = {"color": (255, 255, 255), "thickness": 2}
    default_home_style = {"color": (0, 0, 255), "thickness": 2}
    default_away_style = {"color": (255, 0, 0), "thickness": 2}
    default_save_settings = {"dpi": 100, "fps": 10}

    def __init__(
        self,
        ball_key: str = "ball",
        home_key: str = "0",
        away_key: str = "1",
        ball_style: Optional[Dict] = None,
        home_style: Optional[Dict] = None,
        away_style: Optional[Dict] = None,
        save_settings: Optional[Dict] = None,
    ) -> None:
        self.ball_key = ball_key
        self.home_key = home_key
        self.away_key = away_key
        self.ball_style = {**self.default_ball_style, **(ball_style or {})}
        self.home_style = {**self.default_home_style, **(home_style or {})}
        self.away_style = {**self.default_away_style, **(away_style or {})}
        self.save_settings = {**self.default_save_settings, **(save_settings or {})}

    def _draw_person(self, frame: np.ndarray, box: Rect, style: dict, team_id: str, player_id: str, conf: float) -> None:
        """Draw a single object on the frame."""
        caption = f"{player_id}"
        color = get_color(style["color"])
        center_x, center_y = box.bottom_center.int_xy_tuple

        # draw_rect_with_caption(frame, caption_box, color=color, caption=caption, thickness=style["thickness"])
        draw_ellipse_with_caption(image=frame,rect=box,color=color,thickness=style["thickness"], caption=caption)

    def _draw_ball(self, frame: np.ndarray, box: Rect, style: dict, caption: str) -> None:
        """Draw a single object on the frame."""
        color = Color(*style["color"])
        draw_marker(frame, box, color=color, thickness=style["thickness"])

    def draw_frame(self, frame_df: pd.DataFrame, frame: np.ndarray) -> np.ndarray:
        """Draw bounding boxes from a DataFrame slice onto a frame."""
        frame = frame.copy()
        for (team_id, player_id), player_df in frame_df.iter_players():  # Assuming `iter_players` is a method in frame_df
            if player_df.isnull().any(axis=None):
                continue

            player_df['conf'].values[0]

            conf = player_df['conf'].values[0]

            box = Rect(
                int(player_df['bb_left']),
                int(player_df['bb_top']),
                int(player_df['bb_width']),
                int(player_df['bb_height'])
            )

            if team_id == self.home_key:
                self._draw_person(frame, box, self.home_style, team_id, player_id, conf)
            elif team_id == self.away_key:
                self._draw_person(frame, box, self.away_style , team_id, player_id, conf)
            elif team_id == self.ball_key:
                caption = f"BALL {conf:.2f}%"
                self._draw_ball(frame, box, self.ball_style, caption)

        return frame
