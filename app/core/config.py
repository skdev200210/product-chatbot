"""Application settings, loaded from the environment (and an optional .env file)."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: str = Field(
        description="Anthropic API key. Set ANTHROPIC_API_KEY in the environment."
    )
    model_name: str = Field(
        default="claude-sonnet-5",
        description="Claude model the agent runs on.",
    )
    workflow_base_url: str = Field(
        default="https://productv3workflow.markytics.ai",
        description="Base URL of the workflow API the create_agent tool posts to.",
    )
    max_output_tokens: int = Field(
        default=8192,
        gt=0,
        description=(
            "Output ceiling for one generation. Must comfortably fit a whole template — "
            "hitting it truncates the JSON mid-object."
        ),
    )
    request_timeout_seconds: float = Field(
        default=180.0,
        gt=0,
        description=(
            "A turn that writes a template and then calls create_agent runs two model "
            "round trips and has been measured near 60s, so leave generous headroom."
        ),
    )
    max_conversation_messages: int = Field(default=50, gt=0)
    cors_allow_origins: list[str] = Field(
        default_factory=lambda: ["*"],
        description='Comma-separated in env, e.g. CORS_ALLOW_ORIGINS=["https://app.example.com"]',
    )

    logs_dir: str = Field(default="logs", description="Directory to write log files to.")
    log_level: str = Field(
        default="INFO", description="Root log level: DEBUG, INFO, WARNING, ERROR, CRITICAL."
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]  # values come from the environment


settings = get_settings()