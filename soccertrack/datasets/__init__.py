import os

from soccertrack.logging import logger

from .downloader import KaggleDownloader

__all__ = ["available", "get_path", "KaggleDownloader"]

_module_path = os.path.dirname(__file__)
_available_dir = [p for p in next(os.walk(_module_path))[1] if not p.startswith("__")]
_available_csv = {"soccertrack sample": "soccertrack_sample.csv"}
_available_mp4 = {
    "soccertrack sample": "https://drive.google.com/file/d/1Vxc1NXwLiD3T6cqmlbjgjr-9umDty5Va/view?usp=sharing"
}
available = _available_dir + list(_available_csv.keys())


def get_path(dataset: str, type: str = "csv") -> str:
    """Get the path to the data file.

    Args:
        dataset (str): Name of the dataset. See `soccertrack.datasets.available` for all options.
        dataset_type (str): Type of the dataset. Either 'csv' or 'mp4'.

    Returns:
        str: Path to the data file.
    """
    if type == "csv":
        if dataset in _available_csv:
            fpath = os.path.abspath(os.path.join(_module_path, _available_csv[dataset]))
            return fpath
    if type == "mp4":
        if dataset in _available_mp4:
            fpath = _available_mp4[dataset]
            logger.info(f"Download the dataset from {fpath}")
            return fpath

    msg = f"The dataset '{dataset}' is not available. "
    msg += f"Available datasets are {', '.join(available)}"
    raise ValueError(msg)
