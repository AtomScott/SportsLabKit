"""Customizable logger based on loguru."""

import os
import sys
from collections.abc import Iterable, Mapping
from typing import Any

import __main__ as main
from loguru import logger


class LoggerMixin:
    def __init__(self) -> None:
        """Logger mixin because loguru can't get class names."""

        self.logger = logger.bind(classname=self.__class__.__name__)


def is_interactive() -> bool:
    """True if running in a interactive environment/jupyter notebook.

    Returns:
        bool: True if running in an interactive environment
    """

    return not hasattr(main, "__file__")


def patcher(record: dict[str, str | dict[str, str]]) -> dict[str, str | dict[str, str]]:
    """Customize loguru's log format.

    See the Loguru docs for details on `record` here, https://loguru.readthedocs.io/en/stable/api/logger.html.

    Args:
        record (Dict): Loguru record

    Returns:
        Dict: Loguru record
    """
    if record.get("function") == "<module>":
        if is_interactive():
            record["function"] = "IPython"
        else:
            record["function"] = "Python"

    if record["extra"].get("classname"):
        record["extra"]["classname"] += ":"
    return record


class LevelFilter:
    def __init__(self, level: str = "INFO"):
        """Filter log records based on logging level.

        Args:
            level (str, optional): Logging level to filter on. Defaults to "INFO".
        """
        self._level = level

    def __call__(self, record: Mapping[str, Any]) -> bool:
        """Filter log records based on logging level.

        Args:
            record (Dict): Loguru record

        Returns:
            bool: True if record is at or above the logging level
        """
        levelno = logger.level(self.level).no
        return record["level"].no >= levelno

    @property
    def level(self) -> str:
        """Returns the logging level.

        Returns:
            str: Logging level
        """
        return self._level

    @level.setter
    def level(self, level: str) -> None:
        """Sets the logging level.

        Args:
            level (str): Logging level
        """
        level = level.upper()
        self._level = level


def set_log_level(level: str) -> Any:
    """Set the logging level for the logger.

    Args:
        level (str): Logging level to set
    """
    level_filter.level = level
    os.environ["LOG_LEVEL"] = level


def tqdm(*args, level: str = "INFO", **kwargs) -> Iterable:
    """Wrapper for tqdm.tqdm that uses the logger's level.

    Args:
        *args: Arguments to pass to tqdm.tqdm
        **kwargs: Keyword arguments to pass to tqdm.tqdm
        level (str, optional): Logging level to set. Defaults to "INFO".

    Returns:
        Iterable: Iterable from tqdm progress bar
    """
    from tqdm import tqdm  # noqa

    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    enable = logger.level(LOG_LEVEL).no <= logger.level(level.upper()).no
    kwargs.update({"disable": not enable})
    return tqdm(*args, **kwargs)


def inspect(*args, level: str = "INFO", **kwargs) -> None:
    """Wrapper for rich.inspect that uses the logger's level.

    Args:
        *args: Arguments to pass to rich.inspect
        **kwargs: Keyword arguments to pass to rich.inspect
        level (str, optional): Logging level to set. Defaults to "INFO".
    """
    from rich import inspect

    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
    enable = logger.level(LOG_LEVEL).no <= logger.level(level.upper()).no

    if hasattr(args[0], __name__):
        if args[0].__name__ == "inspect" and enable:
            inspect(inspect, *args[1:], **kwargs)
    elif enable:
        logger.log(level, f"Inspecting: {args}")
        inspect(*args, **kwargs)


def show_df(df, theme="dark"):
    from IPython.display import display

    def dark(styler):
        styler.applymap(lambda x: "color: white")
        styler.set_table_styles(
            [
                {
                    "selector": "th",
                    "props": [("color", "white"), ("background-color", "#555555")],
                }
            ]
        )
        styler.apply(lambda x: ["background: #333333" for _ in x], axis=1)
        return styler

    def light(styler):
        raise NotImplementedError("Light theme not implemented yet.")

    style = dark if theme == "dark" else light
    return display(df.style.pipe(style))


# Code that runs on `import .logger`

logger.remove()
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()


level_filter = LevelFilter(LOG_LEVEL)
config = {
    "handlers": [
        {
            "sink": sys.stdout,
            "format": "<level>{extra[classname]}{function}:{line:04d}  {level.icon}| {message} </level>",
            "colorize": True,
            "filter": level_filter,
        },
    ],
    "levels": [
        {"name": "DEBUG", "color": "<white>", "icon": "🐛"},
        {"name": "INFO", "color": "<cyan>", "icon": "💬"},
        {"name": "SUCCESS", "color": "<green>", "icon": "✅"},
        {"name": "WARNING", "color": "<yellow>", "icon": "🤔"},
        {"name": "ERROR", "color": "<light-red>", "icon": "❌"},
        {"name": "CRITICAL", "color": "<red>", "icon": "🔥"},
    ],
    "patcher": patcher,
    "extra": {"classname": ""},
}

logger.configure(**config)  # type: ignore

if __name__ == "__main__":
    # run `python logger.py` to see the output

    logger.debug("Debug")
    logger.info("Info")
    logger.success("Success")
    logger.warning("Warning")
    logger.error("Error")
    logger.critical("Critical")

    print()

    set_log_level("DEBUG")
    logger.debug("Debug")
    logger.info("Info")
    logger.success("Success")
    logger.warning("Warning")
    logger.error("Error")
    logger.critical("Critical")

    print()

    set_log_level("Critical")
    logger.debug("Debug")
    logger.info("Info")
    logger.success("Success")
    logger.warning("Warning")
    logger.error("Error")
    logger.critical("Critical")
