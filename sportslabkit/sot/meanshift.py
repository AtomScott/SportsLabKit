import cv2
import numpy as np
from sportslabkit.sot.base import SingleObjectTracker


class MeanShiftTracker(SingleObjectTracker):
    required_keys = ["box"]

    def __init__(self, target, initial_frame, bins=16, max_iterations=10, termination_eps=1, *args, **kwargs):
        super().__init__(
            target,
            pre_init_args={
                "initial_frame": initial_frame,
                "bins": bins,
                "max_iterations": max_iterations,
                "termination_eps": termination_eps,
            },
        )

    def pre_initialize(
        self,
        initial_frame,
        bins=16,
        max_iterations=10,
        termination_eps=1,
    ):
        self.bins = bins
        self.max_iterations = max_iterations
        self.termination_eps = termination_eps
        x, y, w, h = self.target["box"]
        self.roi = (x, y, w, h)

        roi_frame = initial_frame[y : y + h, x : x + w]
        hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv_roi, np.array((0.0, 60.0, 32.0)), np.array((180.0, 255.0, 255.0)))

        roi_hist = cv2.calcHist([hsv_roi], [0], mask, [self.bins], [0, 180])
        self.hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

    def update(self, current_frame):
        current_frame = current_frame[0]
        hsv = cv2.cvtColor(current_frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv], [0], self.hist, [0, 180], 1)

        ret, self.roi = cv2.meanShift(
            dst,
            self.roi,
            (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.max_iterations,
                self.termination_eps,
            ),
        )
        x, y, w, h = self.roi
        self.state = {"box": (x, y, w, h)}
        # self.update_tracklet_observations(self.state)
        return self.state

    @property
    def hparam_search_space(self):
        return {
            "bins": {"type": "categorical", "values": [8, 16, 32, 64, 128]},
            "max_iterations": {"type": "categorical", "values": [5, 10, 15, 20]},
            "termination_eps": {"type": "categorical", "values": [1, 2, 3, 4, 5]},
        }
