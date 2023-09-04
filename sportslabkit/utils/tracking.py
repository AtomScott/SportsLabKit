"""Tracking utilities"""


import numpy as np


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
