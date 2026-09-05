"""Path resolution that works identically locally and inside a container.

Hardcoded absolute paths are the usual reason a submission runs on the author's
machine and dies in Docker, so every filesystem path in this project is built
here and nowhere else.

Local run:      <repo root>/data, <repo root>/models
Container run:  /data (bind mount, read-only), /app/models
"""

from __future__ import annotations

import os
from pathlib import Path

# Files that only ever exist at the repository root.
_ROOT_MARKERS = ("requirements.txt", "docker-compose.yml")


def in_container() -> bool:
    """True when running inside Docker.

    /.dockerenv is created by the daemon; DOCKER_CONTAINER is set by our own
    Dockerfile as a belt-and-braces second signal.
    """
    return Path("/.dockerenv").exists() or os.environ.get("DOCKER_CONTAINER") == "1"


def project_root() -> Path:
    """Repository root, found by walking up from this file."""
    here = Path(__file__).resolve()
    for candidate in (here, *here.parents):
        if candidate.is_dir() and any((candidate / m).exists() for m in _ROOT_MARKERS):
            return candidate
    # Fall back to <repo>/src/utils/docker_utils.py -> <repo>
    return here.parents[2]


def data_dir() -> Path:
    """Directory holding the extracted Home Credit CSVs.

    DATA_DIR wins when it names an existing absolute path, which is how the
    container reaches its read-only bind mount and how a local run reaches a
    dataset kept outside the repository.
    """
    configured = os.environ.get("DATA_DIR", "").strip()
    if configured:
        candidate = Path(configured)
        if candidate.is_absolute():
            return candidate
        return project_root() / candidate
    return project_root() / "data"


def models_dir() -> Path:
    """Directory holding serving artifacts. Created if absent."""
    configured = os.environ.get("MODELS_DIR", "").strip()
    path = Path(configured) if configured else project_root() / "models"
    path.mkdir(parents=True, exist_ok=True)
    return path


def data_file(name: str) -> Path:
    """Absolute path to one dataset CSV, verified to exist."""
    path = data_dir() / name
    if not path.exists():
        raise FileNotFoundError(
            f"{name} not found at {path}. "
            f"Extract the Home Credit CSVs and point DATA_DIR at them."
        )
    return path


def config_file(name: str) -> Path:
    return project_root() / "configs" / name
