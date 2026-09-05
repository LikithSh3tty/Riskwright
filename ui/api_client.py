"""HTTP client for the Riskwright API.

The UI's only route to data. Nothing in ui/ imports from src/, so business
logic cannot leak into the presentation layer and the frontend stays
replaceable.
"""

from __future__ import annotations

import os
from typing import Any

import requests

DEFAULT_TIMEOUT = 30
# The chatbot waits on two LLM calls plus a SQL execution.
CHAT_TIMEOUT = 90


class ApiError(RuntimeError):
    """Raised when the API is unreachable or returns a non-2xx response."""


def base_url() -> str:
    return os.environ.get("API_URL", "http://localhost:8000").rstrip("/")


def _request(method: str, path: str, timeout: int, **kwargs: Any) -> Any:
    url = f"{base_url()}{path}"
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.RequestException as exc:
        raise ApiError(f"Could not reach the API at {url}: {exc}") from exc

    if not response.ok:
        detail = response.text.strip()[:300]
        raise ApiError(f"{method} {path} returned {response.status_code}: {detail}")
    return response.json()


def get(path: str, params: dict | None = None, timeout: int = DEFAULT_TIMEOUT) -> Any:
    return _request("GET", path, timeout, params=params)


def post(path: str, payload: dict, timeout: int = DEFAULT_TIMEOUT) -> Any:
    return _request("POST", path, timeout, json=payload)


def health() -> dict:
    return get("/health", timeout=10)
