import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import sys

sys.path.append("../")

from sandbox.functions import *


class TestFunctions(TestCase):
    
        
    def test_cut_video_file(self):
        pass

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
