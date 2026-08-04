"""The pydantic-ai agent that generates call-agent templates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

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

from app.core.config import get_settings
from app.core.prompt import output_variables, render
from app.core.schemas import AgentTemplate, Message, Usage
from app.core.system_prompt import SYSTEM_PROMPT_TEMPLATE
from app.mcp_servers import build_toolsets
from app.core.logger import logger


#: Template fields that may carry ``{{variable}}`` markers. ``summary_prompt``
#: is deliberately absent — it runs after the call, when variables are out of scope.
VARIABLE_FIELDS = ("start_message", "end_message", "objective", "instructions")


class InvalidInputVariables(Exception):
    """`payload["input_variables"]` was not a list of names or a name -> description map."""


@dataclass(frozen=True)
class ChatDeps:
    """Per-request context handed to the agent."""

    system_prompt: str
    allowed_variables: frozenset[str]


def _build_model() -> AnthropicModel:
    settings = get_settings()
    model = AnthropicModel(
        settings.model_name,
        provider=AnthropicProvider(api_key=settings.anthropic_api_key),
    )
    return model


# `NativeOutput` puts the schema on the request itself (`output_config.format`),
# so Anthropic constrains generation to it and the model cannot return prose.
# `max_tokens` matters for that guarantee too: running out mid-object yields
# truncated JSON.
chat_agent = Agent(
    _build_model(),
    deps_type=ChatDeps,
    output_type=NativeOutput(AgentTemplate),
    toolsets=build_toolsets(get_settings().mcp_servers),
    model_settings={"max_tokens": get_settings().max_output_tokens},
    retries=2,
    name="template-generator",
)


@chat_agent.instructions
def _system_prompt(ctx: RunContext[ChatDeps]) -> str:
    return ctx.deps.system_prompt


@chat_agent.output_validator
def _check_variables(ctx: RunContext[ChatDeps], template: AgentTemplate) -> AgentTemplate:
    """Enforce the variable contract, letting the model correct itself.

    Raising ModelRetry sends the problem back to the model rather than failing the
    request, so a hallucinated variable name costs a retry instead of a bad template.
    """
    allowed = ctx.deps.allowed_variables
    problems: list[str] = []

    # Enforced here rather than as `min_length` on the schema, which structured
    # outputs would reject.
    for field in ("start_message", "end_message", "objective", "instructions", "summary_prompt"):
        if not getattr(template, field).strip():
            problems.append(f"{field} is empty.")
    if not template.rules:
        problems.append("rules is empty; give at least one hard constraint.")

    for field in VARIABLE_FIELDS:
        unknown = sorted(output_variables(getattr(template, field)) - allowed)
        if unknown:
            problems.append(f"{field} uses undeclared variables: {', '.join(unknown)}.")

    for index, rule in enumerate(template.rules):
        unknown = sorted(output_variables(rule) - allowed)
        if unknown:
            problems.append(f"rules[{index}] uses undeclared variables: {', '.join(unknown)}.")

    in_summary = sorted(output_variables(template.summary_prompt))
    if in_summary:
        problems.append(
            f"summary_prompt must be plain text but contains variables: {', '.join(in_summary)}."
        )

    if problems:
        available = ", ".join(sorted(allowed)) if allowed else "none"
        raise ModelRetry(
            " ".join(problems)
            + f" The only variables you may use are: {available}."
            + " Rewrite the whole template without the invalid ones."
        )

    return template


def parse_input_variables(payload: Mapping[str, Any]) -> dict[str, str]:
    """Read `input_variables` from the payload as a name -> description mapping.

    Accepts either a list of names or a mapping; a bare list yields empty
    descriptions.
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
    """Render the variable list for the prompt."""
    if not variables:
        return "(none — do not use any variable markers in the template)"
    return "\n".join(
        f"- {{{{{name}}}}}: {description}" if description else f"- {{{{{name}}}}}"
        for name, description in variables.items()
    )


def build_system_prompt(payload: Mapping[str, Any]) -> tuple[str, frozenset[str]]:
    """Render the prompt and return it with the variables the model may use."""
    variables = parse_input_variables(payload)
    prompt = render(
        SYSTEM_PROMPT_TEMPLATE,
        {**payload, "input_variables": format_input_variables(variables)},
    )
    return prompt, frozenset(variables)


def to_message_history(messages: Sequence[Message]) -> list[ModelMessage]:
    """Convert the incoming conversation JSON into pydantic-ai message history."""
    history: list[ModelMessage] = []
    for message in messages:
        if message.role == "user":
            history.append(ModelRequest(parts=[UserPromptPart(content=message.content)]))
        else:
            history.append(ModelResponse(parts=[TextPart(content=message.content)]))
    return history


async def run_chat(
    conversation: Sequence[Message], payload: Mapping[str, Any]
) -> tuple[AgentTemplate, Usage]:
    """Run one turn: last user message is the prompt, everything before it is history."""
    logger.info(
        "running chat_agent with conversation of %d messages and payload: %s",
        len(conversation),
        payload,
    )
    system_prompt, allowed_variables = build_system_prompt(payload)
    result = await chat_agent.run(
        conversation[-1].content,
        message_history=to_message_history(conversation[:-1]),
        deps=ChatDeps(system_prompt=system_prompt, allowed_variables=allowed_variables),
    )

    logger.info(
        "chat_agent output: %s",
        result.output.model_dump_json(indent=2, ensure_ascii=False),
    )
    return result.output, _to_usage(result)


def _to_usage(result: Any) -> Usage:
    # `usage` was a method in older pydantic-ai and is a property now; field names
    # likewise moved from request/response_tokens to input/output_tokens. Accept
    # either shape so a minor upgrade doesn't break the response.
    usage = result.usage
    if callable(usage):
        usage = usage()
    return Usage(
        input_tokens=getattr(usage, "input_tokens", None)
        or getattr(usage, "request_tokens", 0)
        or 0,
        output_tokens=getattr(usage, "output_tokens", None)
        or getattr(usage, "response_tokens", 0)
        or 0,
    )
