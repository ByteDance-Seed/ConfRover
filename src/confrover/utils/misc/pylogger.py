# Copyright (c) 2021 ashleve
# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates.
#
# Original file was released under MIT, with the full license text available in the folder.
#
# This modified file is released under the same license.

from __future__ import annotations

import logging
import sys
from typing import Optional

from lightning.pytorch.utilities import rank_zero_only


def get_pylogger(name=__name__, log_rank_zero_only: bool = True) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    logging_levels = (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    )
    if log_rank_zero_only:
        for level in logging_levels:
            setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


# Common string → logging level map
LEVELS: dict[str, int] = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warn": logging.WARNING,  # allow both "warn" and "warning"
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "fatal": logging.CRITICAL,
    "critical": logging.CRITICAL,
    "notset": logging.NOTSET,
}


class _MaxLevelFilter(logging.Filter):
    """Allow records up to (and including) max_level."""

    def __init__(self, max_level: int) -> None:
        super().__init__()
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D401
        return record.levelno <= self.max_level


def _get_level(level: str | int) -> int:
    """
    Convert string/level to logging level int.
    Examples:
        get_level("info") -> logging.INFO
        get_level("DEBUG") -> logging.DEBUG
        get_level(20) -> 20
    """
    if isinstance(level, int):
        return level
    return LEVELS[level.lower()]


class ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        # 1) use last component of __name__ (logger.name)
        record.shortname = record.name.rsplit(".", 1)[-1]

        # 2) customize level mapping
        custom_levels = {
            "INFO": "",
            "DEBUG": "🔵 ",
            "WARNING": "⚠️ ",
            "ERROR": "❌ ",
            "CRITICAL": "‼️ ",
        }
        record.levelsymbol = custom_levels.get(record.levelname, record.levelname)

        return super().format(record)


def _has_handler(logger: logging.Logger, name: str) -> bool:
    return any(getattr(h, "_name", None) == name for h in logger.handlers)


def set_console_level(level: int, *, logger: Optional[logging.Logger] = None) -> None:
    """Change console handler levels (stdout and/or stderr) together."""
    lg = logger or logging.getLogger("")
    for h in lg.handlers:
        if getattr(h, "_name", "") in {
            "console_stdout",
            "console_stderr",
            "console_stdout_all",
        }:
            # stdout handler keeps DEBUG level to allow MaxLevelFilter; stderr level is WARNING anyway.
            if getattr(h, "_name", "") == "console_stdout_all":
                h.setLevel(level)


def setup_logging(
    *,
    console_level: int | str = logging.INFO,
    split_stderr: bool = False,
    console_format: str = "%(levelsymbol)s%(message)s",
    # datefmt: Optional[str] = None,
    root_name: str = "",  # "" = root logger
    propagate: bool = False,  # set False for library packages
) -> logging.Logger:
    """
    Configure clean console logging once. Safe to call multiple times.

    Parameters
    ----------
    console_level : int
        Logging level for console (stdout/stderr) handlers.
    split_stderr : bool
        If True, INFO and below go to stdout; WARNING and above go to stderr.
        If False, everything goes to stdout.
    console_format : str
        Minimal, readable console formatter. Default: "[I] message".
    datefmt : Optional[str]
        Date format for console. None keeps console timestamp-free by default.
    root_name : str
        Logger name to attach handlers to ("" is the root logger).
    propagate : bool
        If False, prevent double logs when users also configure logging.

    Returns
    -------
    logging.Logger
        The configured base logger.
    """
    base = logging.getLogger(root_name)
    base.setLevel(logging.DEBUG)  # capture everything; handlers filter what to show
    base.propagate = propagate
    console_level = _get_level(console_level)

    # Console formatter: simple and quiet for terminals
    # console_fmt = logging.Formatter(fmt=console_format, datefmt=datefmt)
    console_fmt = ConsoleFormatter(fmt=console_format)

    # stdout handler (INFO and below)
    if split_stderr:
        if not _has_handler(base, "console_stdout"):
            h_out = logging.StreamHandler(stream=sys.stdout)
            h_out.setLevel(logging.DEBUG)  # accept all; filter limits
            h_out.addFilter(_MaxLevelFilter(logging.INFO))
            h_out.setFormatter(console_fmt)
            h_out._name = "console_stdout"
            base.addHandler(h_out)

        # stderr handler (WARNING and above)
        if not _has_handler(base, "console_stderr"):
            h_err = logging.StreamHandler(stream=sys.stderr)
            h_err.setLevel(logging.WARNING)
            h_err.setFormatter(console_fmt)
            h_err._name = "console_stderr"
            base.addHandler(h_err)
    else:
        if not _has_handler(base, "console_stdout_all"):
            h_all = logging.StreamHandler(stream=sys.stdout)
            h_all.setLevel(console_level)
            h_all.setFormatter(console_fmt)
            h_all._name = "console_stdout_all"
            base.addHandler(h_all)

    # Adjust levels after handlers exist
    set_console_level(console_level, logger=base)

    return base


HEADER = "▶"
RESET = "\x1b[0m"
BOLD = "\x1b[1m"


def _supports_color(stream) -> bool:
    return hasattr(stream, "isatty") and stream.isatty()


def _rule(text: str, width: int = 64, char: str = "=") -> str:
    text = f" {text} "
    fill = max(0, width - len(text)) // 2
    return f"{char * fill} {text}{char * fill}"


def log_header(logger: logging.Logger, title: str, char="="):
    stream = next(
        (h.stream for h in logger.handlers if hasattr(h, "stream")), sys.stderr
    )
    line = _rule(title, char=char)
    if _supports_color(stream):
        line = f"\n{BOLD}{line}{RESET}"
    return line


setup_logging(console_level="info")
