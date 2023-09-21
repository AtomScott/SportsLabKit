
from dataclasses import fields
from typing import Type, TypeVar, Union

import pandas as pd

from .slk_dataframe import SLKDataFrame


T = TypeVar("T")

class DataFrameFactory:
    @staticmethod
    def create_empty_dataframe(data_classes: Union[Type[T], list[Type[T]]]) -> SLKDataFrame:
        if not isinstance(data_classes, list):
            data_classes = [data_classes]

        columns = []
        for data_class in data_classes:
            columns += [field.name for field in fields(data_class)]

        df = pd.DataFrame(columns=columns)
        return SLKDataFrame(df)

