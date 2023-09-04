import tempfile
import unittest
from collections import namedtuple
from test.support import captured_stdout
from unittest import mock

import numpy as np

from soccertrack import Camera
from soccertrack.dataframe import BBoxDataFrame
from soccertrack.io.file import load_codf
from soccertrack.logger import *
from soccertrack.types import Detection
from soccertrack.utils import get_git_root

csv_path = (
    get_git_root() / "tests" / "assets" / "codf_sample.csv"
)  # already in pitch coordinates
outputs_path = get_git_root() / "tests" / "outputs"

root = (
    get_git_root()
)  # A 2x2 video with 100 frames and goes from red to green to blue to black in 25 frames each

# A 2x2 video with 100 frames and goes from red to green to blue to black in 25 frames each
rgb_video_path = root / "tests" / "assets" / "videos" / "rgb_video.avi"


class TestBBoxDataFrame(unittest.TestCase):
    def test_test_preprocess_for_mot_eval_1(self):
        """Test for when there are no missing frames"""
        bbdf = BBoxDataFrame.from_dict(
            {
                "home": {
                    "1": {0: [10, 10, 25, 25, 1], 1: [0, 0, 20, 20, 1]},
                    "2": {2: [2, 1, 25, 25, 1]},
                }
            },
            attributes=["bb_left", "bb_top", "bb_width", "bb_height", "conf"],
        )
        ids, dets = bbdf.preprocess_for_mot_eval()

        ans_ids = [np.array([0]), np.array([0]), np.array([1])]
        ans_dets = [
            [np.array([10, 10, 25, 25])],
            [np.array([0, 0, 20, 20])],
            [np.array([2, 1, 25, 25])],
        ]
        for i in range(len(ids)):
            np.testing.assert_almost_equal(ids[i], ans_ids[i])

        for i in range(len(dets)):
            np.testing.assert_almost_equal(dets[i], ans_dets[i])

    def test_preprocess_for_mot_eval_2(self):
        """Test for when there are missing frames"""
        bbdf = BBoxDataFrame.from_dict(
            {
                "home": {
                    "1": {
                        0: [10, 10, 25, 25, 1],
                        2: [5, 0, 25, 25, 1],
                    },
                },
            },
            attributes=["bb_left", "bb_top", "bb_width", "bb_height", "conf"],
        )

        ids, dets = bbdf.preprocess_for_mot_eval()

        ans_ids = [np.array([0]), np.array([]), np.array([0])]
        ans_dets = [
            [np.array([10, 10, 25, 25])],
            [],
            [np.array([5, 0, 25, 25])],
        ]
        for i in range(len(ids)):
            np.testing.assert_almost_equal(ids[i], ans_ids[i])

        for i in range(len(dets)):
            np.testing.assert_almost_equal(dets[i], ans_dets[i])

    def test_test_preprocess_for_mot_eval_3(self):
        """Test for bug that occured when the frames did not start at 0 and has nan value frames."""
        bbdf = BBoxDataFrame.from_dict(
            {
                "home": {
                    "1": {2: [10, 10, 25, 25, 1], 3: [0, 0, 20, 20, 1]},
                }
            },
            attributes=["bb_left", "bb_top", "bb_width", "bb_height", "conf"],
        )
        bbdf = bbdf.reindex(range(1, 4))
        ids, dets = bbdf.preprocess_for_mot_eval()

        ans_ids = [[], np.array([0]), np.array([0])]
        ans_dets = [
            [],
            [np.array([10, 10, 25, 25])],
            [np.array([0, 0, 20, 20])],
        ]

        for i in range(len(ids)):
            np.testing.assert_almost_equal(ids[i], ans_ids[i])

        for i in range(len(dets)):
            np.testing.assert_almost_equal(dets[i], ans_dets[i])

    def test_to_labelbox_data(self):
        """Test for converting to labelbox data"""
        bbdf = BBoxDataFrame.from_dict(
            {
                "home": {
                    "1": {0: [10, 10, 25, 25, 1], 1: [0, 0, 20, 20, 1]},
                    "2": {2: [2, 1, 25, 25, 1]},
                }
            },
            attributes=["bb_left", "bb_top", "bb_width", "bb_height", "conf"],
        )
        MockDataRow = namedtuple("DataRow", ["uid"])
        mock_data_row = MockDataRow("test")

        schema_lookup = {"home_1": "3q4fhvwui45yt", "home_2": "sadfjdhjf1241"}
        with mock.patch("uuid.uuid4", return_value="test_value"):
            data = bbdf.to_labelbox_data(mock_data_row, schema_lookup)

        ans = [
            {
                "uuid": "test_value",
                "schemaId": "3q4fhvwui45yt",
                "dataRow": {"id": "test"},
                "segments": [
                    {
                        "keyframes": [
                            {
                                "frame": 1,
                                "bbox": {
                                    "top": 10,
                                    "left": 10,
                                    "height": 25,
                                    "width": 25,
                                },
                            },
                            {
                                "frame": 2,
                                "bbox": {
                                    "top": 0,
                                    "left": 0,
                                    "height": 20,
                                    "width": 20,
                                },
                            },
                        ]
                    }
                ],
            },
            {
                "uuid": "test_value",
                "schemaId": "sadfjdhjf1241",
                "dataRow": {"id": "test"},
                "segments": [
                    {
                        "keyframes": [
                            {
                                "frame": 3,
                                "bbox": {
                                    "top": 1,
                                    "left": 2,
                                    "height": 25,
                                    "width": 25,
                                },
                            }
                        ]
                    }
                ],
            },
        ]
        self.assertListEqual(data, ans)

    def test_to_labelbox_segment(self):
        """Test for converting to labelbox segment"""
        bbdf = BBoxDataFrame.from_dict(
            {
                "home": {
                    "1": {0: [10, 10, 25, 25, 1], 1: [0, 0, 20, 20, 1]},
                    "2": {2: [2, 1, 25, 25, 1]},
                }
            },
            attributes=["bb_left", "bb_top", "bb_width", "bb_height", "conf"],
        )

        data = bbdf.to_labelbox_segment()

        ans = {
            "home_1": [
                {
                    "keyframes": [
                        {
                            "frame": 1,
                            "bbox": {"top": 10, "left": 10, "height": 25, "width": 25},
                        },
                        {
                            "frame": 2,
                            "bbox": {"top": 0, "left": 0, "height": 20, "width": 20},
                        },
                    ]
                }
            ],
            "home_2": [
                {
                    "keyframes": [
                        {
                            "frame": 3,
                            "bbox": {"top": 1, "left": 2, "height": 25, "width": 25},
                        }
                    ]
                }
            ],
        }
        self.assertDictEqual(data, ans)

    def test_visualize_frame(self):
        """Test for visualizing frame"""
        bbdf = BBoxDataFrame.from_dict(
            {
                "home": {
                    "1": {0: [10, 10, 25, 25, 1]},
                }
            },
            attributes=["bb_left", "bb_top", "bb_width", "bb_height", "conf"],
        )

        cam = Camera(rgb_video_path)
        frame = cam.get_frame(1)

        new_frame = bbdf.visualize_frame(1, frame.copy())
        # Check that the frame is red
        self.assertTrue(np.all(new_frame[0, 0] == [255, 0, 0]))

    def test_visualize_frames(self):
        """Test for visualizing frames"""
        bbdf = BBoxDataFrame.from_dict(
            {
                "home": {
                    "0": {0: [10, 10, 25, 25, 1]},
                    "1": {0: [10, 10, 25, 25, 1]},
                }
            },
            attributes=["bb_left", "bb_top", "bb_width", "bb_height", "conf"],
        )

        # save to temp directory
        with tempfile.TemporaryDirectory() as tmpdirname:
            save_path = os.path.join(tmpdirname, "video.avi")
            bbdf.visualize_frames(rgb_video_path, save_path)

            # check that the video was saved
            self.assertTrue(os.path.exists(save_path))

            # load the video and check that the first frame is red
            cam = Camera(save_path)

            frame = cam.get_frame(0)
            print(frame)
            self.assertTrue(
                np.all(frame[0, 0] == [252, 0, 0])
            )  # value is 252 after compression
