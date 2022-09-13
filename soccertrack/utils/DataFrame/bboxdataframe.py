import pandas as pd
from .base import SoccerTrackMixin


class BBoxDataFrame(SoccerTrackMixin, pd.DataFrame):
    @property
    def _constructor(self):
        return BBoxDataFrame

    # @property
    # def _constructor_sliced(self):
    #     raise NotImplementedError("This pandas method constructs pandas.Series object, which is not yet implemented in {self.__name__}.")

    def to_list(df, xywh=True) -> list:
        """Convert a dataframe column to a 2-dim list for evaluation of object detection.

        Args:
            df (pd.DataFrame): Dataframe
            xywh (bool): If True, convert to x1y1x2y2 format. Defaults to True.

        Returns:
            bbox_2dim_list: 2-dim list
        """
        bbox_2dim_list = []
        bbox_cols = ["bb_left", "bb_top", "bb_width", "bb_height", "conf", "class_id"]
        num_cols = len(bbox_cols)
        df2list = df.values.tolist()

        for frame_id, frame_raw in enumerate(df2list):
            for idx in range(0, len(frame_raw), num_cols):
                bbox_2dim_list.append([frame_id] + frame_raw[idx : idx + num_cols])

        # extract nan rows
        bbox_2dim_list = [x for x in bbox_2dim_list if not any(pd.isnull(x))]
        if xywh:
            for bbox in bbox_2dim_list:
                bbox[1] = int(bbox[1])
                bbox[2] = int(bbox[2])
                bbox[3] = int(bbox[3] + bbox[1])
                bbox[4] = int(bbox[4] + bbox[2])
        else:
            for bbox in bbox_2dim_list:
                bbox[1] = int(bbox[1])
                bbox[2] = int(bbox[2])
                bbox[3] = int(bbox[3])
                bbox[4] = int(bbox[4])
        return bbox_2dim_list
