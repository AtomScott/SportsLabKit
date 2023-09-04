# import datetime
# import os
# import sys
# import unittest
# from pathlib import Path
# from tempfile import TemporaryDirectory
# from unittest import TestCase

import ffmpeg
import pandas as pd

from soccertrack.io.file import (
    load_gps,
    load_gps_from_yaml,
    load_gpsports,
    load_statsports,
)

# TEST_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_DIR = os.path.abspath(os.path.join(TEST_DIR, os.pardir))
# sys.path.insert(0, PROJECT_DIR)

# from sandbox.functions import *  # noqa


#     def test_load_gps_from_yaml(self):
#         yamlfile = Path("tests/assets/gps_data/four_gps_files.yaml")

# def test_cut_video_file(self):
#     """A test for the cut_video_file function."""

#     with TemporaryDirectory() as tmpdir:
#         video_file_name = Path("tests/assets/videos/small-movie.mp4")

#         start_time = 1
#         end_time = 2

#         save_path = Path(tmpdir) / Path("videos/new_video.mp4")
#         cut_video_file(video_file_name, start_time, end_time, save_path)
#         self.assertTrue(save_path.exists())

#         video_info = ffmpeg.probe(str(save_path))["streams"][0]
#         self.assertEqual(video_info["width"], 320)
#         self.assertEqual(video_info["height"], 240)
#         self.assertEqual(
#             float(video_info["duration"]), end_time - start_time
#         )  # value error
#         self.assertEqual(int(video_info["nb_frames"]), (end_time - start_time) * 15)

# def test_load_gpsports(self):
#     filename = Path("tests/assets/gps_data/gpsports/0.xlsx")
#     dataframe = load_gpsports(filename)

#     # assert dataframe shape
#     self.assertEqual(dataframe.shape, (99, 2))

#     # assert headers are correct
#     self.assertListEqual(dataframe.columns.tolist(), [(0, 0, "Lat"), (0, 0, "Lon")])

#     # assert values in first row are correct
#     self.assertListEqual(dataframe.iloc[0].tolist(), [36.10256268, 140.10712484])

#     # assert values in last row are correct
#     self.assertListEqual(dataframe.iloc[-1].tolist(), [36.10256119, 140.10712556])

#     # assert index is correct
#     datetime_value = dataframe.index[0]
#     self.assertEqual(datetime_value, datetime.time(12, 52, 35, 600000))

# def test_load_statsports(self):
#     filename = Path("tests/assets/gps_data/statsports/0.csv")
#     dataframe = load_statsports(filename)

#     # assert dataframe shape
#     self.assertEqual(dataframe.shape, (99, 2))

#     # assert headers are correct
#     self.assertListEqual(dataframe.columns.tolist(), [(0, 0, "Lat"), (0, 0, "Lon")])

#     # assert values in first row are correct
#     self.assertListEqual(dataframe.iloc[0].tolist(), [36.102613, 140.107261])

#     # assert values in last row are correct
#     self.assertListEqual(dataframe.iloc[-1].tolist(), [36.10257933, 140.1071105])

#     # assert index is correct
#     datetime_value = dataframe.index[0]
#     self.assertEqual(datetime_value, datetime.time(12, 48, 47, 800000))

# def test_load_gps_with_single_gpsports(self):
#     filename = Path("tests/assets/gps_data/gpsports/0.xlsx")
#     dataframe = load_gps(filename)
#     pd.testing.assert_frame_equal(dataframe, load_gpsports(filename))

# def test_load_gps_with_single_statsports(self):
#     filename = Path("tests/assets/gps_data/statsports/0.csv")
#     dataframe = load_gps(filename)
#     pd.testing.assert_frame_equal(dataframe, load_statsports(filename))

# def test_load_gps_with_multiple_formats(self):
#     filenames = [
#         Path("tests/assets/gps_data/gpsports/0.xlsx"),
#         Path("tests/assets/gps_data/gpsports/1.xlsx"),
#         Path("tests/assets/gps_data/statsports/0.csv"),
#         Path("tests/assets/gps_data/statsports/1.csv"),
#     ]
#     dataframe = load_gps(filenames)

#     # TODO: add assertions

# def test_load_gps_from_yaml(self):
#     yamlfile = Path("tests/assets/gps_data/four_gps_files.yaml")

#     dataframe = load_gps_from_yaml(yamlfile)

# TODO: add assertions

# def test_cut_gps_file(self):
#     """A test for the cut_gps_file function."""
#     with TemporaryDirectory() as tmpdir:
#         video_file_name = Path("tests/assets/small-gps-data.csv")

#         start_time = 1
#         end_time = 2

#         save_path = Path(tmpdir) / Path("videos/new_video.mp4")
#         cut_video_file(video_file_name, start_time, end_time, save_path)
#         self.assertTrue(save_path.exists())

#         video_info = ffmpeg.probe(str(save_path))["streams"][0]
#         self.assertEqual(video_info["width"], 320)
#         self.assertEqual(video_info["height"], 240)
#         self.assertEqual(
#             float(video_info["duration"]), end_time - start_time
#         )  # value error
#         self.assertEqual(int(video_info["nb_frames"]), (end_time - start_time) * 15)
#     pass

# def test_visualization_gps(self):
#     pass

# def test_visualization_annotations(self):
#     pass

# def test_upload2s3(self):
#     pass

# def test_download_from_s3(self):
#     pass

# def test_upload_annotation2labelbox(self):
#     pass

# def test_upload_video2labelbox(self):
#     pass

# def test_create_annotation_df_from_s3(self):
#     pass


# if __name__ == "__main__":
#     unittest.main()
