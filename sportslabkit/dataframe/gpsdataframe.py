import pandas as pd

from .base import BaseSLKDataFrame


class GPSDataFrame(BaseSLKDataFrame, pd.DataFrame):
    @property
    def _constructor(self):
        return GPSDataFrame

    # @property
    # def _constructor_sliced(self):
    #     raise NotImplementedError("This pandas method constructs pandas.Series object, which is not yet implemented in {self.__name__}.")
