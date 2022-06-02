import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import sys
import ffmpeg

from .sandbox.functions import *  # noqa


class TestFunctions(TestCase):
    """test functions"""

    def test_cut_video_file(self):
        """A test for the cut_video_file function."""

        with TemporaryDirectory() as tmpdir:
            video_file_name = Path("tests/assets/videos/small-movie.mp4")

            start_time = 1
            end_time = 2

            save_path = Path(tmpdir) / Path("videos/new_video.mp4")
            cut_video_file(video_file_name, start_time, end_time, save_path)
            self.assertTrue(save_path.exists())

            video_info = ffmpeg.probe(str(save_path))["streams"][0]
            self.assertEqual(video_info["width"], 320)
            self.assertEqual(video_info["height"], 240)
            self.assertEqual(
                float(video_info["duration"]), end_time - start_time
            )  # value error
            self.assertEqual(int(video_info["nb_frames"]), (end_time - start_time) * 15)

    def test_load_gpsports(self):
        filename = Path("tests/assets/gps_data/xxx")
        dataframe = load_gpsports(filename)

        answer_dict = {"latitude": [1, 2, 3], "longitude": [1, 2, 3]}
        self.assertDictEqual(dataframe.to_dict(), answer_dict)

    def test_load_statsports(self):
        filename = Path("tests/assets/gps_data/xxx")
        dataframe = load_statsports(filename)

        answer_dict = {"latitude": [1, 2, 3], "longitude": [1, 2, 3]}
        self.assertDictEqual(dataframe.to_dict(), answer_dict)

    def test_load_gps(self):
        filename = Path("tests/assets/gps_data/xxx")
        dataframe = load_gpsports(filename)

        answer_dict = {"latitude": [1, 2, 3], "longitude": [1, 2, 3]}
        self.assertDictEqual(dataframe.to_dict(), answer_dict)

        filename = Path("tests/assets/gps_data/xxx")
        dataframe = load_gpsports(filename)

        answer_dict = {"latitude": [1, 2, 3], "longitude": [1, 2, 3]}
        self.assertDictEqual(dataframe.to_dict(), answer_dict)
    
    def test_load_gps_from_yaml(self):
        yamlfile = Path("tests/assets/gps_data/test_players_22.yaml")
        pass
    
    def test_cut_gps_file(self):
        """A test for the cut_gps_file function."""
        with TemporaryDirectory() as tmpdir:
            video_file_name = Path("tests/assets/small-gps-data.csv")

            start_time = 1
            end_time = 2

            save_path = Path(tmpdir) / Path("videos/new_video.mp4")
            cut_video_file(video_file_name, start_time, end_time, save_path)
            self.assertTrue(save_path.exists())

            video_info = ffmpeg.probe(str(save_path))["streams"][0]
            self.assertEqual(video_info["width"], 320)
            self.assertEqual(video_info["height"], 240)
            self.assertEqual(
                float(video_info["duration"]), end_time - start_time
            )  # value error
            self.assertEqual(int(video_info["nb_frames"]), (end_time - start_time) * 15)
        pass

    def test_visualization_gps(self):
        pass

    def test_visualization_annotations(self):
        pass

    def test_upload2s3(self):
        pass

    def test_download_from_s3(self):
        pass

    def test_upload_annotation2labelbox(self):
        pass

    def test_upload_video2labelbox(self):
        pass

    def test_create_annotation_df_from_s3(self):
        pass


if __name__ == "__main__":
    unittest.main()
