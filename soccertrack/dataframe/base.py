from abc import abstractmethod
import pandas as pd
from ast import literal_eval
from typing import Any, Union

import dateutil.parser
import numpy as np
import pandas as pd
from pandas._typing import FilePath, WriteBuffer

class SoccerTrackMixin(object):

    def save_dataframe(
        self,
        path_or_buf: Union[FilePath, WriteBuffer[bytes], WriteBuffer[str]],
    ) -> None:
        """Save a dataframe to a file.

        Args:
            df (pd.DataFrame): Dataframe to save.
            path_or_buf (FilePath | WriteBuffer[bytes] | WriteBuffer[str]): Path to save the dataframe.
        """
        assert self.columns.nlevels == 3, "Dataframe must have 3 levels of columns"
        if self.attrs:
            # write dataframe attributes to the csv file
            with open(path_or_buf, "w") as f:
                for k, v in self.attrs.items():
                    f.write(f"#{k}:{v}\n")
            self.to_csv(path_or_buf, mode="a")
        else:
            self.to_csv(path_or_buf, mode="w")