
import pandas as pd
from .base import SoccerTrackMixin

class BBoxDataFrame(SoccerTrackMixin, pd.DataFrame):

    @property
    def _constructor(self):
        return BBoxDataFrame

    # @property
    # def _constructor_sliced(self):
    #     raise NotImplementedError("This pandas method constructs pandas.Series object, which is not yet implemented in {self.__name__}.")