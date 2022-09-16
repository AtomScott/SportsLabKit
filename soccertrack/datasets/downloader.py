from __future__ import annotations

import os
import platform
from pathlib import Path
from typing import Any, Optional, Union

import click

from soccertrack.logging import inspect, logger

_pathlike = Union[str, Path]
_module_path = os.path.dirname(__file__)


class KaggleDownloader:
    def __init__(self) -> None:
        """An downloader to download the soccertrack dataset from Kaggle."""
        self.api = authenticate()
        self.dataset_owner = "atomscott"
        self.dataset_name = "soccertrack"

    def show_info(self) -> None:
        """Show the dataset info."""
        dataset = self.api.dataset_list(
            search=f"{self.dataset_owner}/{self.dataset_name}"
        )[0]
        inspect(dataset)

    def dataset_list_files(self) -> None:
        """List the files in the dataset."""
        files = self.api.dataset_list_files(f"{self.dataset_owner}/{self.dataset_name}")
        logger.info(files)

    def download(
        self,
        file_name: Optional[str] = None,
        path: Optional[_pathlike] = None,
        force: bool = False,
        quiet: bool = True,
        unzip: bool = False,
    ) -> None:
        """Download the dataset from Kaggle.

        Args:
            file_name (Optional[str], optional): Name of the file to download. If None, downloads all data. Defaults to None.
            path (Optional[_pathlike], optional): Path to download the data to. If None, downloads to soccertrack/datasets/data. Defaults to None.
            force (bool, optional): If True, overwrites the existing file. Defaults to False.
            quiet (bool, optional): If True, suppresses the output. Defaults to True.
            unzip (bool, optional): If True, unzips the file. Defaults to False.
        """

        if file_name is None:
            self.api.dataset_download_files(
                f"{self.dataset_owner}/{self.dataset_name}",
                path=path,
                force=force,
                quiet=quiet,
                unzip=unzip,
            )
        else:
            self.api.dataset_download_files(
                f"{self.dataset_owner}/{self.dataset_name}",
                path=path,
                force=force,
                quiet=quiet,
            )


def get_platform() -> str:
    """Get the platform of the current operating system.

    Returns:
        str: The platform of the current operating system, one of "linux", "mac", "windows", "other".
    """

    platforms = {
        "linux": "linux",
        "linux1": "linux",
        "linux2": "linux",
        "darwin": "mac",
        "win32": "windows",
    }

    if platform.system().lower() not in platforms:
        return "other"

    return platforms[platform.system().lower()]


def show_authenticate_message() -> Any:
    """Show the instructions to authenticate the Kaggle API key."""
    logger.info("Please authenticate with your Kaggle account.")
    has_account = click.confirm("\tDo you have a Kaggle account?", abort=False)

    if has_account:
        platform = get_platform()
        username = click.prompt("\tPlease enter your kaggle username", type=str)
        print(
            f"\tPlease go to https://www.kaggle.com/{username}/account and follow these steps:"
        )
        print('\t\t1. Scroll and click the "Create API Token" section.')
        print('\t\t2. A file named "kaggle.json" will be downloaded.')

        if platform in ["linux", "mac"]:
            print("\t\t3. Move the file to ~/.kaggle/kaggle.json")
        elif platform == "windows":
            print(
                "\t\t3. Move the file to C:\\Users\\<Windows-username>\\.kaggle\\kaggle.json"
            )
        else:
            print(
                "\t\t3. Move the file to ~/.kaggle/kaggle.json  folder in Mac and Linux or to C:\\Users\\<Windows-username>\\.kaggle\\kaggle.json  on windows."
            )

        click.confirm(
            "\tHave you completed the steps above? Type N to abort", abort=True
        )

        return authenticate(show_message=False)

    print(
        "\tPlease create a Kaggle account and follow the instructions on the following:"
    )
    print("\thttps://www.kaggle.com/")
    return None


def authenticate(show_message: bool = True) -> Any:
    """Authenticate the Kaggle API key."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # noqa
    except OSError:
        logger.error("Kaggle API key not found. Showing instructions to authenticate.")
        if show_message:
            return show_authenticate_message()
        return None

    api = KaggleApi()
    api.authenticate()
    return api


if __name__ == "__main__":
    downloader = KaggleDownloader()
    downloader.show_info()
