"""Application settings, loaded from the environment (and an optional .env file)."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.core.models import MCPServerConfig


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
    mcp_servers: list[MCPServerConfig] = Field(
        default_factory=list,
        description=(
            "Remote MCP servers the agent may call, as a JSON list in MCP_SERVERS. "
            "See MCPServerConfig in app/core/models.py for the entry shape."
        ),
    )
    max_output_tokens: int = Field(
        default=8192,
        gt=0,
        description=(
            "Output ceiling for one generation. Must comfortably fit a whole template — "
            "hitting it truncates the JSON mid-object."
        ),
    )
    request_timeout_seconds: float = Field(default=60.0, gt=0)
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