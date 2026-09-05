"""Path resolution tests.

Hardcoded paths that work locally and break in the container are the classic
way a submission like this fails on the evaluator's machine, so the resolution
rules get tests rather than trust.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils import docker_utils
from src.utils.helpers import to_identifier


def test_project_root_holds_the_root_markers():
    root = docker_utils.project_root()
    assert (root / "requirements.txt").exists() or (root / "docker-compose.yml").exists()


def test_absolute_data_dir_is_used_verbatim(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    assert docker_utils.data_dir() == tmp_path


def test_relative_data_dir_resolves_against_project_root(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "data")
    assert docker_utils.data_dir() == docker_utils.project_root() / "data"


def test_unset_data_dir_falls_back_to_repo_data(monkeypatch):
    monkeypatch.delenv("DATA_DIR", raising=False)
    assert docker_utils.data_dir() == docker_utils.project_root() / "data"


def test_missing_dataset_file_names_the_path_and_the_fix(monkeypatch, tmp_path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    with pytest.raises(FileNotFoundError) as excinfo:
        docker_utils.data_file("application_train.csv")
    message = str(excinfo.value)
    assert "application_train.csv" in message
    assert "DATA_DIR" in message


def test_models_dir_is_created_on_demand(monkeypatch, tmp_path):
    target = tmp_path / "artifacts"
    monkeypatch.setenv("MODELS_DIR", str(target))
    assert docker_utils.models_dir() == target
    assert target.is_dir()


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("SK_ID_CURR", "sk_id_curr"),
        ("AMT_INCOME_TOTAL", "amt_income_total"),
        ("  DAYS_BIRTH  ", "days_birth"),
        ("FLAG-OWN-CAR", "flag_own_car"),
        ("2_YEAR_FLAG", "c_2_year_flag"),
    ],
)
def test_headers_normalise_to_unquoted_postgres_identifiers(raw, expected):
    # Postgres folds unquoted identifiers to lowercase. Normalising on load
    # means generated SQL never needs quoting, which removes a whole class of
    # LLM error.
    assert to_identifier(raw) == expected


def test_blank_header_is_rejected():
    with pytest.raises(ValueError):
        to_identifier("   ")


def test_in_container_is_false_locally(monkeypatch):
    monkeypatch.delenv("DOCKER_CONTAINER", raising=False)
    if not Path("/.dockerenv").exists():
        assert docker_utils.in_container() is False
