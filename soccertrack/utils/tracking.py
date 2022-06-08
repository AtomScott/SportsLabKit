"""Tracking utilities"""

import numpy as np
from copy import deepcopy
from filterpy.common import Saver

# large floating point number
inf = 1.0e10


class GatedEuclideanDistance:
    def __init__(self, feature="xy", min_limit=-inf, max_limit=inf):
        self.feature = feature
        self.min_limit = min_limit
        self.max_limit = max_limit

    def __call__(self, x, y):
        distance = np.linalg.norm(x - y)
        if distance < self.min_limit:
            return inf
        if distance > self.max_limit:
            return inf
        return distance


def cost(X, Y, func):
    feature = func.feature
    x_feature = getattr(X, "_feature_" + feature)
    y_feature = getattr(Y, "_feature_" + feature)
    return func(x_feature, y_feature)


def cdist(detections, tracks, funcs, reduction="mean"):

    costs = []
    for func in funcs:
        C = np.zeros((len(detections), len(tracks)))

        for di, detection in enumerate(detections):
            for ti, track in enumerate(tracks):
                C[di, ti] = cost(detection, track.detections[-1], func=func)
        costs.append(C)

    if reduction == "mean":
        return np.mean(costs, axis=0)
    else:
        raise NotImplementedError(f"reduction(`{reduction}`) not implemented")


class Track:
    def __init__(self, initial_detection, kf=None, funcs=[]):
        self.kf = kf

        self.saver = Saver(kf)
        self._detections = [initial_detection]
        self._funcs = funcs

    def predict(self):
        self.kf.predict()

    def update(self, candidate_detection):
        candidate_detection = deepcopy(candidate_detection)
        xy = np.array([candidate_detection.px, candidate_detection.py])
        self.kf.update(xy)

        self._detections.append(candidate_detection)
        self.save()

    def save(self):
        self.saver.save()
        pass

    def associate(self, detections, return_cost=False):
        cost_vec = cdist(detections, [self], self._funcs)

        if len(cost_vec) == 0:
            if return_cost:
                return None, cost_vec
            return None
        if return_cost:
            return detections[cost_vec.argmin()], cost_vec
        return detections[cost_vec.argmin()]

    @property
    def detections(self):
        return self._detections

    @property
    def _feature_xy(self):
        if len(self.detections) == 0:
            return np.array([self.initial_px, self.initial_py])
        return np.array([self.detections[-1].px, self.detections[-1].py])

    @property
    def _feature_motion(self):
        pass

    @property
    def _feature_appearance(self):
        pass
