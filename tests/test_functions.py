import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import sys
import ffmpeg

sys.path.append("../")

from sandbox.functions import * # noqa


class TestFunctions(TestCase):
    """ test functions """
    
    def test_cut_video_file(self):
        """A test for the cut_video_file function."""

        with TemporaryDirectory() as tmpdir:
            video_file_name = Path('../assets/small-movie.mp4')

            start_time = 1
            end_time = 2
            
            save_path = Path(tmpdir) / Path("videos/new_video.mp4")
            cut_video_file(video_file_name, start_time, end_time, save_path)
            self.assertTrue(save_path.exists())
            
            video_info = ffmpeg.probe(str(save_path))['streams'][0]
            self.assertEqual(video_info['width'], 320)
            self.assertEqual(video_info['height'], 240)
            self.assertEqual(int(video_info['duration']), end_time - start_time)
            self.assertEqual(int(video_info['nb_frames']), (end_time - start_time)*25)


    def test_cut_gps_file(self):
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
