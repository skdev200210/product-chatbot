"""MCP server access for the agent.

Servers are declared in one environment variable, ``MCP_SERVERS``, holding a JSON
list. Only remote HTTP/SSE servers are supported — the app does not spawn local
stdio processes.

Each entry carries its own tool allowlist, so a server can advertise a hundred
tools while the agent only ever sees the handful you approved::

    MCP_SERVERS='[
      {"name": "docs", "url": "https://mcp.example.com/mcp",
       "auth": "sk-...", "allowed_tools": ["search_docs", "get_page"]},
      {"name": "internal", "url": "https://tools.internal/mcp"}
    ]'

Omitting ``allowed_tools`` exposes every tool the server advertises.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence

from app.core.models import MCPServerConfig
from app.core.logger import logger


if TYPE_CHECKING:  # pragma: no cover
    from pydantic_ai.toolsets import AbstractToolset


def apply_allowlist(
    toolset: AbstractToolset[Any], allowed_tools: list[str] | None
) -> AbstractToolset[Any]:
    """Restrict a toolset to ``allowed_tools``; ``None`` leaves it untouched."""
    if allowed_tools is None:
        return toolset

    allowed = frozenset(allowed_tools)
    return toolset.filtered(lambda ctx, tool_def: tool_def.name in allowed)


def build_toolsets(servers: Sequence[MCPServerConfig]) -> list[AbstractToolset[Any]]:
    """Turn the configured servers into toolsets the agent can be given.

    Nothing connects here — sessions open when the agent is entered (see the
    app lifespan in ``app/main.py``).
    """
    if not servers:
        return []

    try:
        from pydantic_ai.mcp import MCPToolset
    except ImportError as exc:  # pragma: no cover - depends on how deps were installed
        raise RuntimeError(
            "MCP servers are configured but MCP support is not installed. "
            'Install it with: uv add "pydantic-ai-slim[anthropic,mcp]"'
        ) from exc

    toolsets: list[AbstractToolset[Any]] = []
    for server in servers:
        toolset = apply_allowlist(
            MCPToolset(
                server.url,
                id=server.name,
                auth=server.auth,
                headers=server.headers or None,
            ),
            server.allowed_tools,
        )

        logger.info(
            "mcp server %r at %s (tools: %s)",
            server.name,
            server.url,
            ", ".join(server.allowed_tools) if server.allowed_tools is not None else "all",
        )
        toolsets.append(toolset)

    return toolsets
