
import pandas as pd

from .core_functions import visualize_boxes, visualize_points


class SLKDataFrame(pd.DataFrame):
    """Sportslabkit DataFrame class extending pandas DataFrame."""

    def visualize_boxes(self) -> None:
        """Delegate the visualize_boxes operation to the core function."""
        visualize_boxes(self)

    def visualize_points(self) -> None:
        """Delegate the visualize_points operation to the core function."""
        visualize_points(self)
