"""Per-turn handoff of the agent config to the remote MCP server.

The MCP `create_agent` tool needs the config ids from the request payload, and those
ids deliberately never reach the model. Rather than carry them as a tool argument,
each turn writes them here under a per-turn id and sends only that id to the server.

Keys are per-turn, never per-conversation. The payload is not stable across a
conversation — the client echoes the previous template back into it, so turn 3 of a
conversation carries different values than turn 1 — and a shared key would hand one
turn's config to another.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

import redis.asyncio as redis

from app.core.config import get_settings
from app.core.logger import logger

_KEY_PREFIX = "agent-chatbot:mcp-ctx:"

# Echoed back by the client from the turn before. The model passes the current
# template as tool arguments, so storing these too would hand the server a stale
# second copy of the template with no rule for which one wins.
_ECHOED_TEMPLATE_KEYS = frozenset(
    {"start_msg", "end_msg", "objectives", "instructions", "rules", "summary_prompt"}
)

_client: redis.Redis | None = None


async def connect() -> None:
    """Open the connection pool. Called once at startup."""
    global _client
    if _client is not None:
        return
    settings = get_settings()
    client = redis.from_url(settings.redis_url, decode_responses=True)
    await client.ping()
    _client = client
    logger.info("redis connected, mcp context ttl=%ss", settings.mcp_context_ttl_seconds)


async def close() -> None:
    global _client
    if _client is not None:
        await _client.aclose()
        _client = None
        logger.info("redis connection closed")


def key(context_id: str) -> str:
    """The key the MCP server reads this turn's config from."""
    return f"{_KEY_PREFIX}{context_id}"


async def put(context_id: str, payload: Mapping[str, Any]) -> None:
    """Store one turn's config for the MCP server to read.

    Raises if the pool was never opened or Redis is unreachable — the caller decides
    whether that is fatal. The key is never deleted after being read: `retries=2`
    means `create_agent` can fire more than once in a turn and each attempt re-reads
    it, so it is left to expire on its TTL instead.
    """
    if _client is None:
        raise RuntimeError("redis pool is not open — connect() was not called at startup")

    config = {name: value for name, value in payload.items()}
    await _client.set(
        key(context_id),
        json.dumps(config, default=str),
        ex=get_settings().mcp_context_ttl_seconds,
    )
    logger.info(
        "stored mcp context %s: %d keys kept, %d echoed template keys dropped",
        context_id,
        len(config),
        len(payload) - len(config),
    )
