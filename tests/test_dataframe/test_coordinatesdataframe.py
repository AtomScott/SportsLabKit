import unittest
from test.support import captured_stdout

from soccertrack.logger import *
from soccertrack.dataframe import CoordinatesDataFrame
from soccertrack.io.file import load_codf
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

    def test_visualize_frames(self):
        codf = load_codf(csv_path)
        save_path = outputs_path / "test_visualize_frames.mp4"

        # make sure the file does not exist or delete it if it does
        if save_path.exists():
            save_path.unlink()

        codf.visualize_frames(save_path=save_path)
        assert save_path.exists(), f"File {save_path} does not exist"

    def test_numpy(self):
        pass

    def test_to_pitch_coordinates(self):
        pass
