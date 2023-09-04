from __future__ import annotations

import platform
from pathlib import Path
from typing import Any, Optional, Union

from sportslabkit.logger import inspect, logger


_pathlike = Union[str, Path]
_module_path = Path(__file__).parent


class KaggleDownloader:
    def __init__(self) -> None:
        """An downloader to download the soccertrack dataset from Kaggle."""
        self.api = authenticate()
        self.dataset_owner = "atomscott"
        self.dataset_name = "soccertrack"

    def show_info(self) -> None:
        """Show the dataset info."""
        dataset = self.api.dataset_list(search=f"{self.dataset_owner}/{self.dataset_name}")[0]
        inspect(dataset)

    def dataset_list_files(self) -> None:
        """List the files in the dataset."""
        files = self.api.dataset_list_files(f"{self.dataset_owner}/{self.dataset_name}")
        logger.info(files)

    def download(
        self,
        file_name: Optional[str] = None,
        path: Optional[_pathlike] = _module_path,
        force: bool = False,
        quiet: bool = False,
        unzip: bool = True,
    ) -> None:
        """Download the dataset from Kaggle.

        Args:
            file_name (Optional[str], optional): Name of the file to download. If None, downloads all data. Defaults to None.
            path (Optional[_pathlike], optional): Path to download the data to. If None, downloads to soccertrack/datasets/data. Defaults to None.
            force (bool, optional): If True, overwrites the existing file. Defaults to False.
            quiet (bool, optional): If True, suppresses the output. Defaults to True.
            unzip (bool, optional): If True, unzips the file. Defaults to True.
        """

        path = Path(path)
        if file_name is None:
            self.api.dataset_download_files(
                f"{self.dataset_owner}/{self.dataset_name}",
                path=path,
                force=force,
                quiet=quiet,
                unzip=unzip,
            )
        else:
            self.api.dataset_download_file(
                f"{self.dataset_owner}/{self.dataset_name}",
                file_name=file_name,
                path=path,
                force=force,
                quiet=quiet,
            )

        if file_name is None and unzip:
            file_name = "soccertrack"
        if file_name is None and not unzip:
            file_name += "soccertrack.zip"
        else:
            file_name = Path(file_name).name

        save_path = path / file_name
        return save_path


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


def confirm(msg: str) -> bool:
    """Confirm the user input."""
    logger.info(msg + " [y/n]")
    val = input()
    logger.info(f"You entered: {val}")
    if val.lower() in ["y", "yes"]:
        return True
    elif val.lower() in ["n", "no"]:
        return False
    else:
        logger.error("Invalid input. Please try again.")
        return confirm(msg)


def prompt(msg: str, type: Any) -> Any:
    """Prompt the user for input."""
    logger.info(msg)
    val = input()
    logger.info(f"You entered: {val}")
    try:
        return type(val)
    except ValueError:
        logger.error("Invalid input. Please try again.")
        return prompt(msg, type)


def show_authenticate_message() -> Any:
    """Show the instructions to authenticate the Kaggle API key."""
    logger.info("Please authenticate with your Kaggle account.")
    has_account = confirm("Do you have a Kaggle account?")

    if has_account:
        platform = get_platform()
        username = prompt("Please enter your kaggle username", type=str)
        logger.info(f"Please go to https://www.kaggle.com/{username}/account and follow these steps:")
        logger.info('1. Scroll and click the "Create API Token" section.')
        logger.info('2. A file named "kaggle.json" will be downloaded.')

        if platform in ["linux", "mac"]:
            logger.info("3. Move the file to ~/.kaggle/kaggle.json")
        elif platform == "windows":
            logger.info("3. Move the file to C:\\Users\\<Windows-username>\\.kaggle\\kaggle.json")
        else:
            logger.info(
                "3. Move the file to ~/.kaggle/kaggle.json  folder in Mac and Linux or to C:\\Users\\<Windows-username>\\.kaggle\\kaggle.json  on windows."
            )

        if not confirm("Have you completed the steps above? Type N to abort."):
            logger.info("Aborting.")
            return None

        return authenticate(show_message=False)

    logger.info("Please create a Kaggle account and follow the instructions on the following:")
    logger.info("https://www.kaggle.com/")
    return None


def authenticate(show_message: bool = True) -> Any:
    """Authenticate the Kaggle API key."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi  # noqa

        api = KaggleApi()
        api.authenticate()
        logger.info("Authentication successful.")
    except OSError:
        logger.error("Kaggle API key not found. Showing instructions to authenticate.")
        if show_message:
            return show_authenticate_message()
        return None

    return api


if __name__ == "__main__":
    downloader = KaggleDownloader()
    downloader.show_info()
