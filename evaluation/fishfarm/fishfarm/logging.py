"""
Copied from Optuna repo:
https://github.com/optuna/optuna/blob/2595653638506e1b7e025a966a220984a59ab936/optuna/logging.py
Removed some comments for less verbosity.

In general, `logger.info` is preferred over `print` since it contains module name and timestamp;
We recommend the use of logger object for the fishfarm developers.

Inside fishfarm, we can call `get_logger(__name__)` from each python file.
Then the root logger format and level are applied to that logger object.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
from logging import CRITICAL, DEBUG, ERROR, FATAL, INFO, WARN, WARNING

import colorlog


__all__ = [
    "CRITICAL",
    "DEBUG",
    "ERROR",
    "FATAL",
    "INFO",
    "WARN",
    "WARNING",
]

_lock: threading.Lock = threading.Lock()
_default_handler: logging.Handler | None = None


def create_default_formatter() -> logging.Formatter:
    """Create a default formatter of log messages.

    This function is not supposed to be directly accessed by library users.
    """
    header = "[%(levelname)1.1s %(asctime)s %(name)s]"
    message = "%(message)s"
    if _color_supported():
        return colorlog.ColoredFormatter(
            f"%(log_color)s{header}%(reset)s {message}",
        )
    return logging.Formatter(f"{header} {message}")


def _color_supported() -> bool:
    """Detection of color support."""
    # NO_COLOR environment variable:
    if os.environ.get("NO_COLOR", None):
        return False

    if not hasattr(sys.stderr, "isatty") or not sys.stderr.isatty():
        return False
    else:
        return True


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if _default_handler:
            # This library has already configured the library root logger.
            return
        _default_handler = logging.StreamHandler()  # Set sys.stderr as stream.
        _default_handler.setFormatter(create_default_formatter())

        # Apply our default configuration to the library root logger.
        library_root_logger: logging.Logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(logging.INFO)
        library_root_logger.propagate = False


def _reset_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if not _default_handler:
            return

        library_root_logger: logging.Logger = _get_library_root_logger()
        library_root_logger.removeHandler(_default_handler)
        library_root_logger.setLevel(logging.NOTSET)
        _default_handler = None


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the specified name.
    name's prefix should be `fishfarm.` (just like __name__ variable),
    otherwise root logger settings will be not reflected.
    This function is not supposed to be directly accessed by library users.
    """

    _configure_library_root_logger()
    return logging.getLogger(name)


def get_verbosity() -> int:
    """Return the current level for the fishfarm's root logger.

    Returns:
        Logging level, e.g., ``fishfarm.logging.DEBUG`` and ``fishfarm.logging.INFO``.

    .. note::
        fishfarm has following logging levels:

        - ``fishfarm.logging.CRITICAL``, ``fishfarm.logging.FATAL``
        - ``fishfarm.logging.ERROR``
        - ``fishfarm.logging.WARNING``, ``fishfarm.logging.WARN``
        - ``fishfarm.logging.INFO``
        - ``fishfarm.logging.DEBUG``
    """

    _configure_library_root_logger()
    return _get_library_root_logger().getEffectiveLevel()


def set_verbosity(verbosity: int) -> None:
    """Set the level for the fishfarm's root logger.

    Args:
        verbosity:
            Logging level, e.g., ``fishfarm.logging.DEBUG`` and ``fishfarm.logging.INFO``.

    .. note::
        fishfarm has following logging levels:

        - ``fishfarm.logging.CRITICAL``, ``fishfarm.logging.FATAL``
        - ``fishfarm.logging.ERROR``
        - ``fishfarm.logging.WARNING``, ``fishfarm.logging.WARN``
        - ``fishfarm.logging.INFO``
        - ``fishfarm.logging.DEBUG``
    """

    _configure_library_root_logger()
    _get_library_root_logger().setLevel(verbosity)


def disable_default_handler() -> None:
    """Disable the default handler of the fishfarm's root logger."""

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().removeHandler(_default_handler)


def enable_default_handler() -> None:
    """Enable the default handler of the fishfarm's root logger."""

    _configure_library_root_logger()

    assert _default_handler is not None
    _get_library_root_logger().addHandler(_default_handler)


def disable_propagation() -> None:
    """Disable propagation of the library log outputs.

    Note that log propagation is disabled by default. You only need to use this function
    to stop log propagation when you use :func:`~fishfarm.logging.enable_propagation()`.
    """

    _configure_library_root_logger()
    _get_library_root_logger().propagate = False


def enable_propagation() -> None:
    """Enable propagation of the library log outputs.

    Please disable the fishfarm's default handler to prevent double logging if the root logger has
    been configured.
    """

    _configure_library_root_logger()
    _get_library_root_logger().propagate = True
