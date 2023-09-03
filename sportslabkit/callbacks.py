
"""Defines the Callback base class and utility decorators for use with the Trainer class.

The Callback class provides a dynamic way to hook into various stages of the Trainer's operations.
It uses Python's __getattr__ method to dynamically handle calls to methods that are not explicitly defined,
allowing it to handle arbitrary `on_<event_name>_start` and `on_<event_name>_end` methods.

Example:
    class MyPrintingCallback(Callback):
        def on_train_start(self, trainer):
            print("Training is starting")
"""

from functools import wraps


def with_callbacks(func):
    """
    Decorator for wrapping methods that require callback invocations.

    Args:
        func (callable): The method to wrap.

    Returns:
        callable: The wrapped method.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        event_name = func.__name__
        self._invoke_callbacks(f'on_{event_name}_start')
        result = func(self, *args, **kwargs)
        self._invoke_callbacks(f'on_{event_name}_end')
        return result
    return wrapper


class Callback:
    """Base class for creating new callbacks.

    This class defines the basic structure of a callback and allows for dynamic method creation
    for handling different events in the Trainer's lifecycle.

    Methods:
        __getattr__(name: str) -> callable:
            Returns a dynamically created method based on the given name.
    """

    pass
