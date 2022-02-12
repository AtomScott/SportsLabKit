from math import cos, radians, sin
from filterpy.common.kinematic import kinematic_state_transition

import numpy as np
from filterpy.kalman import KalmanFilter
from filterpy.common import kinematic_kf


def kinematic_kf_ord1(P, Q, R, dt):
    """Creates a constant acceleration Kalman filter for 2d space."""

    dim = 2
    order = 1

    kf = kinematic_kf(dim=dim, order=3, dt=dt)

    F = np.eye(kf.F.shape[0])
    for ndim in range(0, dim):
        F[
            ndim * (order + 1) : (ndim + 1) * (order + 1),
            ndim * (order + 1) : (ndim + 1) * (order + 1),
        ] = kinematic_state_transition(order, dt)

    kf.P *= P
    kf.F = F

    kf.Q *= Q
    kf.R *= R
    return kf


def kinematic_kf_ord2(P, Q, R, dt):
    """Creates a constant acceleration Kalman filter for 2d space."""

    dim = 2
    order = 2
    kf = kinematic_kf(dim=dim, order=3, dt=dt)

    F = np.eye(kf.F.shape[0])
    for ndim in range(0, dim):
        F[
            ndim * (order + 1) : (ndim + 1) * (order + 1),
            ndim * (order + 1) : (ndim + 1) * (order + 1),
        ] = kinematic_state_transition(order, dt)

    kf.P *= P
    kf.F = F

    kf.Q *= Q
    kf.R *= R
    return kf


def kinematic_kf_ord3(P, Q, R, dt):
    """Creates a constant acceleration Kalman filter for 2d space."""

    kf = kinematic_kf(dim=2, order=3, dt=dt)
    kf.P *= P
    kf.Q *= Q
    kf.R *= R
    return kf
