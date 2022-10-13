
from contextlib import contextmanager
import logging

@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    previous_level = logging.root.manager.disable
    logging.disable(highest_level)
    try:
        yield
    finally:
        logging.disable(previous_level)

# load vidgear first with all_logging_disabled
with all_logging_disabled():
    from vidgear.gears import WriteGear, CamGear

from soccertrack.dataframe import CoordinatesDataFrame, BBoxDataFrame
from soccertrack.io import load_df
import soccertrack.datasets  # noqa
from soccertrack.camera import Camera
