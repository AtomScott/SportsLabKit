import logging
import warnings
from contextlib import contextmanager


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)


# load vidgear first with all_logging_disabled
with all_logging_disabled(), warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from vidgear.gears import WriteGear, CamGear

import soccertrack.datasets  # noqa
from soccertrack.camera import Camera
from soccertrack.dataframe import BBoxDataFrame, CoordinatesDataFrame
from soccertrack.io import load_df
