"""The chat agent: writes voice-agent templates, and creates them on request.

`create_agent` is a plain tool that calls the workflow API directly — the same
two POSTs the MCP server made. Doing it here means no MCP server to reach, and
the ids come from the request payload rather than from the model, so it cannot
invent or mistype them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from fastmcp.client import Client
from fastmcp.client.transports import StreamableHttpTransport
from pydantic_ai.mcp import MCPToolset

import httpx
from pydantic import ValidationError
from pydantic_ai import Agent, ModelRetry, NativeOutput, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.capabilities import MCP

from app.core import context_store
from app.core.config import get_settings
from app.core.logger import logger
from app.core.prompt import render
from app.core.schemas import AgentCreated, AgentTemplate, Message, Usage
from app.core.system_prompt import SYSTEM_PROMPT_TEMPLATE

import uuid


class InvalidInputVariables(Exception):
    """`payload["input_variables"]` was not a list of names or a name -> description map."""


# The model sometimes stubs the long prose fields when filling in the tool call —
# `"instructions": "placeholder"` — and that text is what the created agent runs on.
# Catch it here so a stub is never posted to the workflow API.
_STUB_VALUES = frozenset(
    {
        "placeholder",
        "todo",
        "tbd",
        "n/a",
        "na",
        "none",
        "null",
        "string",
        "text",
        "text here",
        "same as above",
        "as above",
        "see above",
        "unchanged",
        "...",
    }
)

# A floor low enough that no genuinely written field trips it; "placeholder" is 11 chars.
_MIN_FIELD_LENGTH = 25


def _stub_fields(template: AgentTemplate) -> list[str]:
    """Names of template fields that were left as a stub rather than actually written."""
    stubs = []
    for name in ("start_message", "end_message", "objective", "instructions", "summary_prompt"):
        value = getattr(template, name).strip()
        if value.strip("<>[[]().").lower() in _STUB_VALUES or len(value) < _MIN_FIELD_LENGTH:
            stubs.append(name)
    if not template.rules or any(
        rule.strip().strip("<>[[]().").lower() in _STUB_VALUES for rule in template.rules
    ):
        stubs.append("rules")
    return stubs


@dataclass
class ChatDeps:
    """Per-request context. `agent_config` never reaches the model."""

    system_prompt: str
    agent_config: Mapping[str, Any]
    # Names this turn's slot in Redis, and travels to the MCP server as a header.
    context_id: str
    # Set by create_agent on success, so a created turn can still return its template.
    created_template: AgentTemplate | None = None


async def inject_agent_config(
    ctx: RunContext[ChatDeps],
    call_tool: Any,
    name: str,
    args: dict[str, Any],
) -> Any:
    """Hand the request's config to the MCP server out of band, not through the model.

    The remote `create_agent` needs the config ids (client_id, product_id, llm_id, ...)
    but `agent_config` deliberately never reaches the model, so the model cannot supply
    them. They go into Redis under this turn's context id, which the server already has
    from the request header — keeping the ids out of the prompt and out of reach of a
    mistyped or invented value.

    The write happens here rather than at the top of the turn because most turns only
    write a template and call no tool at all; there is nothing to store until the tool
    actually fires. Writing before `call_tool` means the key is always in place by the
    time the server looks for it.
    """
    if name == "create_agent":
        config = ctx.deps.agent_config
        if not config:
            logger.warning("create_agent called but payload.agent_config was empty")
            return {"error": "No agent_config was supplied with this request, so nothing can be created."}

        # The response keeps the template a created agent was built from, and the tool
        # arguments are the only place it exists — the model returns AgentCreated instead.
        try:
            ctx.deps.created_template = AgentTemplate.model_validate(
                {k: v for k, v in args.items() if k != "payload"}
            )
        except ValidationError as exc:
            logger.warning("could not capture template from create_agent args: %s", exc)

        try:
            await context_store.put(ctx.deps.context_id, config)
            logger.info("create_agent context stored for %r", config.get("agent_name"))
        except Exception:
            # Redis being down should not fail the turn while the server still accepts
            # the inline argument. Drop this fallback once the server reads only Redis.
            logger.exception("could not store mcp context, falling back to inline payload")
            args = {**args, "payload": dict(config)}

    result = await call_tool(name, args)
    logger.info("%s returned: %s", name, str(result)[:400])
    return result


def parse_input_variables(payload: Mapping[str, Any]) -> dict[str, str]:
    """Read `input_variables` as a name -> description mapping.

    Accepts a list of names or a mapping; a bare list yields empty descriptions.
    """
    raw = payload.get("input_variables")
    if raw is None:
        return {}
    if isinstance(raw, Mapping):
        return {str(name): str(description).strip() for name, description in raw.items()}
    if isinstance(raw, (list, tuple, set)):
        return {str(name): "" for name in raw}
    raise InvalidInputVariables(
        "payload.input_variables must be a list of names or a mapping of name -> description, "
        f"got {type(raw).__name__}"
    )


def format_input_variables(variables: Mapping[str, str]) -> str:
    if not variables:
        return "(none — do not use any variable markers in the template)"
    return "\n".join(
        f"- {{{{{name}}}}}: {description}" if description else f"- {{{{{name}}}}}"
        for name, description in variables.items()
    )


def build_system_prompt(payload: Mapping[str, Any]) -> str:
    """Fill the prompt's <<key>> markers from the request payload."""
    return render(
        SYSTEM_PROMPT_TEMPLATE,
        {**payload, "input_variables": format_input_variables(parse_input_variables(payload))},
    )


def to_message_history(messages: Sequence[Message]) -> list[ModelMessage]:
    history: list[ModelMessage] = []
    for message in messages:
        if message.role == "user":
            history.append(ModelRequest(parts=[UserPromptPart(content=message.content)]))
        else:
            history.append(ModelResponse(parts=[TextPart(content=message.content)]))
    return history


async def run_chat(
    conversation: Sequence[Message], payload: Mapping[str, Any]
) -> tuple[AgentTemplate | None, AgentCreated | None, Usage]:
    """Run one turn: last user message is the prompt, everything before it is history.

    Returns the template for the turn and, when the agent was actually created, the
    creation result alongside it — a created turn keeps the template it was built from.
    """
    logger.info(
        "turn: %d message(s), payload keys=%s, agent_name=%s",
        len(conversation),
        sorted(payload),
        payload.get("agent_name"),
    )
    context_id = str(uuid.uuid4())
    logger.info("turn context_id=%s", context_id)

    deps = ChatDeps(
        system_prompt=build_system_prompt(payload),
        agent_config=payload,
        context_id=context_id,
    )

    chat_agent = Agent(
        AnthropicModel(
            get_settings().model_name,
            provider=AnthropicProvider(api_key=get_settings().anthropic_api_key),
        ),
        deps_type=ChatDeps,
        output_type=NativeOutput(
            [AgentTemplate, AgentCreated],
            name="chat_turn",
            description=(
                "AgentTemplate when writing or revising a template. "
                "AgentCreated only after create_agent has actually run."
            ),
        ),
        model_settings={"max_tokens": get_settings().max_output_tokens},
        retries=0,
        name="template-generator",
    )

    # Built per turn so the header carries this turn's id; the server reads the config
    # from Redis under it when create_agent runs.
    mcp_client = Client(
        StreamableHttpTransport(
            get_settings().mcp_server_url,
            headers={"X-Context-ID": context_id},
        )
    )
    turn_toolset = MCPToolset(mcp_client, process_tool_call=inject_agent_config)

    @chat_agent.instructions
    def _system_prompt(ctx: RunContext[ChatDeps]) -> str:
        return ctx.deps.system_prompt

    result = await chat_agent.run(
        conversation[-1].content,
        message_history=to_message_history(conversation[:-1]),
        deps=deps,
        toolsets=[turn_toolset],
    )

    tools_called = [
        part.tool_name
        for message in result.all_messages()
        for part in message.parts
        if part.part_kind == "tool-call"
    ]
    logger.info("turn produced %s, tools called=%s", type(result.output).__name__, tools_called or "none")

    output = result.output
    if isinstance(output, AgentTemplate):
        template, created = output, None
    else:
        # The model only returns AgentCreated after create_agent ran, so the template it
        # was created from is the one the tool captured.
        created = output
        template = deps.created_template
        if template is None:
            logger.warning("turn returned AgentCreated but no template was captured")

    usage = result.usage
    if callable(usage):  # property in current pydantic-ai, method in older releases
        usage = usage()
    return template, created, Usage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
    )
