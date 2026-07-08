"""Application configuration, loaded from environment variables and .env."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    exa_api_key: SecretStr = Field(
        description="Exa API key from https://dashboard.exa.ai"
    )
    exa_base_url: str = "https://api.exa.ai"
    exa_timeout_seconds: float = 20.0

    # If unset, the Anthropic SDK falls back to the ANTHROPIC_API_KEY env
    # var or an `ant auth login` profile.
    anthropic_api_key: SecretStr | None = None
    evaluation_model: str = "claude-opus-4-8"
    evaluation_effort: str = "medium"
    evaluation_timeout_seconds: float = 120.0

    # Public-demo protections; all inactive unless demo_mode is true.
    demo_mode: bool = False
    daily_request_budget: int = 120
    ip_requests_per_hour: int = 4
    ip_requests_per_day: int = 8
    # Only set true behind a proxy you control (reads X-Forwarded-For).
    trust_proxy_headers: bool = False

    # Optional usage recording; everything stays off until a path is set.
    usage_db_path: str | None = None  # SQLite file, e.g. /data/usage.db
    usage_salt: SecretStr | None = None  # keeps visitor hashes stable across restarts
    stats_token: SecretStr | None = None  # enables GET /stats when set
