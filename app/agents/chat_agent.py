"""The chat agent: writes voice-agent templates, and creates them on request.

`create_agent` is a plain tool that calls the workflow API directly — the same
two POSTs the MCP server made. Doing it here means no MCP server to reach, and
the ids come from the request payload rather than from the model, so it cannot
invent or mistype them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import httpx
from pydantic_ai import Agent, NativeOutput, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider

from app.core.config import get_settings
from app.core.logger import logger
from app.core.prompt import render
from app.core.schemas import AgentCreated, AgentTemplate, Message, Usage
from app.core.system_prompt import SYSTEM_PROMPT_TEMPLATE

ChatOutput = AgentTemplate | AgentCreated


class InvalidInputVariables(Exception):
    """`payload["input_variables"]` was not a list of names or a name -> description map."""


@dataclass(frozen=True)
class ChatDeps:
    """Per-request context. `agent_config` never reaches the model."""

    system_prompt: str
    agent_config: Mapping[str, Any]


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
    retries=2,
    name="template-generator",
)


@chat_agent.instructions
def _system_prompt(ctx: RunContext[ChatDeps]) -> str:
    return ctx.deps.system_prompt


@chat_agent.tool
async def create_agent(ctx: RunContext[ChatDeps], template: AgentTemplate) -> dict[str, Any]:
    """Create the voice agent from a template.

    Call this when the user asks for the agent to be created, saved, or deployed.
    Pass the finished template; the account settings come from the request.
    """
    config = ctx.deps.agent_config
    if not config:
        logger.warning("create_agent called but payload.agent_config was empty")
        return {"error": "No agent_config was supplied with this request, so nothing can be created."}

    base = get_settings().workflow_base_url
    logger.info("creating agent %r", config.get("agent_name"))

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            master_payload = {
                "agent_name": config.get("agent_name"),
                "client_id": int(config.get("client_id")),
                "product_id": int(config.get("product_id")),
                "channel_id": int(config.get("channel_id")),
            }
            logger.info("create_agent master payload: %s", master_payload)
            master = await client.post(
                f"{base}/api/v1/agent-master",
                json=master_payload,
            )

            logger.info("create_agent master response: %s %s", master.status_code, master.text)

            master.raise_for_status()
            master_id = master.json().get("agent_id")

            master_agent_payload = {
                "agent_name": config.get("agent_name"),
                "agent_master_id": int(master_id),
                "client_id": int(config.get("client_id")),
                "product_id": int(config.get("product_id")),
                "channel_id": int(config.get("channel_id")),
                "parent_agent_id": 0,
                "input_file_id": int(config.get("input_file_id")),
                "start_msg": template.start_message,
                "end_msg": template.end_message,
                "instructions": template.instructions,
                "rules": "\n".join(template.rules),
                "objectives": template.objective,
                "summary_prompt": template.summary_prompt,
                "is_global_agent": False,
                "languages_supported": [int(lang) for lang in config.get("languages_supported", [])],
                "language_names": config.get("languages"),
                # "language_id": 0,
                # "language_name": "string",
                "llm_id": int(config.get("llm_id")),
                "stt_id": int(config.get("stt_id")),
                "tts_id": int(config.get("tts_id")),
                "llm_name": config.get("llm_name"),
                "stt_name": config.get("stt_name"),
                "tts_name": config.get("tts_name"),
                "voice": config.get("voice"),
                "channel_type": "AI_CALL",
                "use_calling_config_defaults": False,
                "expected_input_columns": config.get("expected_input_columns"),
                "expected_output_columns": config.get("expected_output_columns"),
                "is_start_template": True,
            }
            logger.info("create_agent created payload: %s", master_agent_payload)

            created = await client.post(
                f"{base}/api/v1/agents",
                json=master_agent_payload,
            )

            logger.info("create_agent created response: %s %s", created.status_code, created.text)
            created.raise_for_status()
    except httpx.HTTPStatusError as exc:
        # Hand the failure to the model so it tells the user instead of claiming success.
        logger.error("create_agent failed: %s %s", exc.response.status_code, exc.response.text)
        return {"error": f"{exc.response.status_code}: {exc.response.text[:400]}"}
    except httpx.HTTPError as exc:
        logger.error("create_agent could not reach %s: %s", base, exc)
        return {"error": f"could not reach the workflow API: {exc}"}

    logger.info("created agent %r", config.get("agent_name"))
    return {"message": "Agent created successfully", "agent_name": config.get("agent_name")}


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
) -> tuple[ChatOutput, Usage]:
    """Run one turn: last user message is the prompt, everything before it is history."""
    logger.info(
        "turn: %d message(s), payload keys=%s, agent_name=%s",
        len(conversation),
        sorted(payload),
        payload.get("agent_name"),
    )
    result = await chat_agent.run(
        conversation[-1].content,
        message_history=to_message_history(conversation[:-1]),
        deps=ChatDeps(
            system_prompt=build_system_prompt(payload),
            agent_config=payload,
        ),
    )

    tools_called = [
        part.tool_name
        for message in result.all_messages()
        for part in message.parts
        if part.part_kind == "tool-call"
    ]
    logger.info("turn produced %s, tools called=%s", type(result.output).__name__, tools_called or "none")

    usage = result.usage
    if callable(usage):  # property in current pydantic-ai, method in older releases
        usage = usage()
    return result.output, Usage(
        input_tokens=getattr(usage, "input_tokens", 0) or 0,
        output_tokens=getattr(usage, "output_tokens", 0) or 0,
    )
