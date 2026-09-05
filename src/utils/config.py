"""Central configuration, read once from the environment.

Every tunable lives here so that nothing downstream reads os.environ directly.
Defaults are chosen so the app runs under docker-compose with no .env edits
beyond DATA_DIR and the Anthropic key.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Tables loaded into Postgres. This tuple is the single source of truth: the
# loader creates exactly these, and the SQL validator whitelists exactly these.
ALLOWED_TABLES: tuple[str, ...] = (
    "application_train",
    "bureau",
    "previous_application",
)

# Hard ceiling on rows any chatbot-generated query may return.
MAX_QUERY_ROWS = 200

# Wall-clock budget for a single chatbot query.
QUERY_TIMEOUT_SECONDS = 10


def _env_file() -> Path:
    """Absolute path to .env, resolved from this file rather than the cwd.

    A relative ".env" is resolved against the working directory, so running
    anything from a subdirectory (a notebook, a script) silently picks up the
    default passwords instead of the real ones and fails to authenticate. The
    repository root is found by walking up from here.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "requirements.txt").exists():
            return candidate / ".env"
    return here.parents[2] / ".env"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Postgres, owner role. Used by the loader and by read-only app queries
    # that are not chatbot-generated.
    postgres_db: str = "riskwright"
    postgres_user: str = "riskwright"
    postgres_password: str = "change_me"
    postgres_host: str = "postgres"
    postgres_port: int = 5432

    # Postgres, read-only role. Used exclusively by the chatbot query runner.
    postgres_ro_user: str = "riskwright_ro"
    postgres_ro_password: str = "change_me_too"

    # Data
    data_dir: str = "data"
    force_reload: bool = False

    # LLM
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-haiku-4-5"

    # Services
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_url: str = "http://api:8000"
    log_level: str = Field(default="INFO")

    def dsn(self, read_only: bool = False) -> str:
        """libpq connection string.

        read_only selects the restricted role, which holds SELECT and nothing
        else. Anything touching chatbot-generated SQL must pass read_only=True.
        """
        user = self.postgres_ro_user if read_only else self.postgres_user
        password = self.postgres_ro_password if read_only else self.postgres_password
        return (
            f"host={self.postgres_host} port={self.postgres_port} "
            f"dbname={self.postgres_db} user={user} password={password}"
        )

    @property
    def llm_configured(self) -> bool:
        return bool(self.anthropic_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
