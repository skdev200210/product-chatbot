from pydantic import BaseModel, ConfigDict, Field, field_validator


class MCPServerConfig(BaseModel):
    """One remote MCP server."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        min_length=1,
        description="Identifier used in logs and to distinguish servers. Not sent to the model.",
    )
    url: str = Field(description="Streamable-HTTP or SSE endpoint of the MCP server.")
    auth: str | None = Field(
        default=None,
        description="Bearer token. Sent as `Authorization: Bearer <auth>`.",
    )
    headers: dict[str, str] = Field(
        default_factory=dict, description="Extra headers sent on every request to this server."
    )
    allowed_tools: list[str] | None = Field(
        default=None,
        description="Tool names the agent may use. Omit or set null to allow every tool.",
    )

    @field_validator("url")
    @classmethod
    def _must_be_http(cls, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            raise ValueError("MCP server url must start with http:// or https://")
        return url
