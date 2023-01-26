import unittest
from test.support import captured_stdout

import numpy as np

from soccertrack.dataframe import CoordinatesDataFrame
from soccertrack.io.file import load_codf
from soccertrack.logger import *
from soccertrack.types import Detection
from soccertrack.utils import get_git_root

csv_path = (
    get_git_root() / "tests" / "assets" / "codf_sample.csv"
)  # already in pitch coordinates
outputs_path = get_git_root() / "tests" / "outputs"


class TestCoordinatesDataFrame(unittest.TestCase):
    def test_load(self):
        codf = load_codf(csv_path)
        self.assertIsInstance(codf, CoordinatesDataFrame)

    def test_visualize_frame(self):
        codf = load_codf(csv_path)
        save_path = outputs_path / "test_visualize_frame.png"

        # make sure the file does not exist or delete it if it does
        if save_path.exists():
            save_path.unlink()

        codf.visualize_frame(0, save_path=save_path)
        assert save_path.exists(), f"File {save_path} does not exist"

    def test_visualize_frame_with_custom_kwargs(self):
        codf = load_codf(csv_path)
        save_path = outputs_path / "test_visualize_frame_with_custom_save_kwargs.png"

        # make sure the file does not exist or delete it if it does
        if save_path.exists():
            save_path.unlink()

        marker_kwargs = {"markerfacecolor": "green", "ms": 30}
        saved_kwargs = {"dpi": 300, "bbox_inches": "tight"}

        print(codf)

        codf.visualize_frame(
            0,
            save_path=save_path,
            marker_kwargs=marker_kwargs,
            save_kwargs=saved_kwargs,
        )
        assert save_path.exists(), f"File {save_path} does not exist"

    def test_visualize_frames(self):
        codf = load_codf(csv_path)
        save_path = outputs_path / "test_visualize_frames.mp4"

        # make sure the file does not exist or delete it if it does
        if save_path.exists():
            save_path.unlink()

        codf.visualize_frames(save_path=save_path)
        assert save_path.exists(), f"File {save_path} does not exist"

    def test_visualize_frames_with_custom_save_kwargs(self):
        codf = load_codf(csv_path)
        save_path = outputs_path / "test_visualize_frames_with_custom_save_kwargs.mp4"

        # make sure the file does not exist or delete it if it does
        if save_path.exists():
            save_path.unlink()

        saved_kwargs = {"dpi": 300, "fps": 50}

        codf.visualize_frames(save_path=save_path, save_kwargs=saved_kwargs)
        assert save_path.exists(), f"File {save_path} does not exist"

    def test_from_numpy_1(self):
        arr = np.random.rand(10, 22, 2)
        codf = CoordinatesDataFrame.from_numpy(arr)

        self.assertIsInstance(codf, CoordinatesDataFrame)
        self.assertEqual(codf.shape, (10, 44))

    def test_from_numpy_2(self):
        arr = np.random.rand(10, 23, 2)
        codf = CoordinatesDataFrame.from_numpy(arr)

        self.assertIsInstance(codf, CoordinatesDataFrame)
        self.assertEqual(codf.shape, (10, 46))

    def test_from_numpy_3(self):
        arr = np.random.rand(5, 3, 2)
        team_ids = [1, 2, "ball"]
        player_ids = [1, 1, "ball"]
        codf = CoordinatesDataFrame.from_numpy(arr, team_ids, player_ids)

        assert codf.shape == (5, 6)

    def test_from_dict_1(self):
        d = {
            "home_team": {
                "player_1a": {1: (1, 2), 2: (3, 4), 3: (5, 6)},
                "player_1b": {1: (7, 8), 2: (9, 10), 3: (11, 12)},
            },
            "away_team": {
                "player_2a": {1: (13, 14), 2: (15, 16), 3: (17, 18)},
                "player_2b": {1: (19, 20), 2: (21, 22), 3: (23, 24)},
            },
            "ball": {"ball": {1: (25, 26), 2: (27, 28), 3: (29, 30)}},
        }
        codf = CoordinatesDataFrame.from_dict(d)

        assert codf.shape == (3, 10)

    def test_from_dict_2(self):
        d = {
            "home_team": {
                "player_1a": {1: (1, 2), 2: (3, 4), 3: (5, 6), 4: (5, 6), 5: (5, 6)},
                "player_1b": {1: (7, 8), 2: (9, 10), 3: (11, 12)},
            },
            "away_team": {
                "player_2a": {1: (13, 14), 2: (15, 16), 3: (17, 18)},
                "player_2b": {1: (19, 20), 2: (21, 22), 3: (23, 24)},
            },
            "ball": {"ball": {3: (29, 30)}},
        }
        codf = CoordinatesDataFrame.from_dict(d)
        print(codf)
        assert codf.shape == (5, 10)

    def test_to_pitch_coordinates(self):
        pass  # TODO
