import unittest
from test.support import captured_stdout

from soccertrack.logger import *
from soccertrack.tracking_model import SingleObjectTracker
from soccertrack.types import Detection

det0 = Detection(box=[10, 10, 5, 5], score=0.9, class_id=0, feature=[1, 2, 3])
det1 = Detection(box=[20, 20, 3, 3], score=0.75, class_id=0, feature=[1, 1, 0])


class TestSingleObjectTracker(unittest.TestCase):
    def test_init(self):
        sot = SingleObjectTracker()

    def test_update(self):
        sot = SingleObjectTracker()
        sot.update(detection=det0)

        self.assertEqual(sot.box, det0.box)
        self.assertEqual(sot.score, det0.score)
        self.assertEqual(sot.class_id, det0.class_id)
        self.assertEqual(sot.feature, det0.feature)

        sot.update(detection=det1)

        self.assertEqual(sot.box, det1.box)
        self.assertEqual(sot.score, det1.score)
        self.assertEqual(sot.class_id, det1.class_id)
        self.assertEqual(sot.feature, det1.feature)

        sot.update(detection=None)
        self.assertEqual(sot.box, det1.box)
        self.assertEqual(sot.score, det1.score)
        self.assertEqual(sot.class_id, det1.class_id)
        self.assertEqual(sot.feature, det1.feature)

        self.assertEqual(sot.steps_positive, 2)
        self.assertEqual(sot.steps_alive, 3)

    def test_staleness(self):
        sot = SingleObjectTracker(max_staleness=2)
        sot.update(detection=det0)
        sot.update(detection=det1)
        sot.update(detection=None)
        sot.update(detection=None)
        sot.update(detection=None)

        self.assertEqual(sot.steps_positive, 2)
        self.assertEqual(sot.steps_alive, 5)
        self.assertEqual(sot.staleness, 3)
        self.assertTrue(sot.is_stale)

    def test_print(self):
        sot = SingleObjectTracker()
        sot.update(detection=det0)
        sot.update(detection=det1)
        sot.update(detection=None)
        sot.update(detection=None)
        sot.update(detection=None)

        with captured_stdout() as stdout:
            print(sot)
        self.assertEqual(
            stdout.getvalue(),
            "(box: [20, 20, 3, 3], score: 0.75, class_id: 0, staleness: 3.00)\n",
        )

    def test_to_bbdf(self):
        sot = SingleObjectTracker()
        sot.update(detection=det0)
        sot.update(detection=det1)
        sot.update(detection=None)

        df = sot.to_bbdf()
        self.assertEqual(df.shape, (3, 5))
        self.assertEqual(df.iloc[0].tolist(), [10.0, 10.0, 5.0, 5.0, 0.9])
        self.assertEqual(df.iloc[1].tolist(), [20.0, 20.0, 3.0, 3.0, 0.75])
        self.assertEqual(df.iloc[2].tolist(), [20.0, 20.0, 3.0, 3.0, 0.75])
