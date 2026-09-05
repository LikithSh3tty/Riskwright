"""Logging setup.

One configured root handler, obtained through get_logger. Modules never call
logging.basicConfig themselves, which would fight over the root handler when
uvicorn and Streamlit both start.
"""

from __future__ import annotations

import logging
import sys

_CONFIGURED = False

_FORMAT = "%(asctime)s %(levelname)-7s %(name)s | %(message)s"
_DATEFMT = "%H:%M:%S"


def _configure() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    # Imported lazily so logging works even if settings fail to load.
    from src.utils.config import get_settings

    try:
        level = get_settings().log_level.upper()
    except Exception:  # pragma: no cover - configuration is not worth crashing on
        level = "INFO"

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATEFMT))

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(name)
