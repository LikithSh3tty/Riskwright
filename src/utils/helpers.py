"""Small shared utilities.

Kept deliberately thin. Anything that grows a second responsibility moves out
into its own module.
"""

from __future__ import annotations

import re
import time
from contextlib import contextmanager
from typing import Iterator

from src.utils.logger import get_logger

log = get_logger(__name__)

# Postgres folds unquoted identifiers to lowercase, so every column is
# lowercased on load and generated SQL never needs quoting.
_IDENT_CLEAN = re.compile(r"[^0-9a-z_]")


def to_identifier(name: str) -> str:
    """Normalise a CSV header into a safe, unquoted Postgres identifier."""
    ident = _IDENT_CLEAN.sub("_", name.strip().lower())
    ident = re.sub(r"_+", "_", ident).strip("_")
    if not ident:
        raise ValueError(f"column name {name!r} normalises to nothing")
    if ident[0].isdigit():
        ident = f"c_{ident}"
    return ident


@contextmanager
def timed(label: str) -> Iterator[None]:
    """Log how long a slow step took. Used on data loads and training runs."""
    start = time.perf_counter()
    log.info("%s: started", label)
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log.info("%s: finished in %.1fs", label, elapsed)


def human_count(n: int) -> str:
    """1234567 -> '1,234,567'."""
    return f"{n:,}"


def safe_divide(numerator: float, denominator: float) -> float | None:
    """Ratio features divide by columns that are legitimately zero or null."""
    if denominator in (0, None) or numerator is None:
        return None
    return numerator / denominator
