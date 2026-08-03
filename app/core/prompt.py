"""Rendering the system prompt template.

Two different placeholder syntaxes are in play, and keeping them apart matters:

``<<key>>``
    Filled in *by this app* from the request payload before the prompt is sent
    to the model. See ``app/core/system_prompt.py``.

``{{variable}}``
    Left alone here. These belong to the template the agent *generates*, and are
    substituted later by whatever runs the call. The agent is told which ones it
    may use; enforcement lives in the output validator in
    ``app/agents/chat_agent.py``.

Angle brackets are used for our own markers so the prompt can talk about
``{{variable}}`` literally, and so that JSON examples, code snippets and ``$``
need no escaping.
"""

from __future__ import annotations

import json
import re
from typing import Any, Mapping

_PLACEHOLDER_RE = re.compile(r"<<\s*([A-Za-z_][A-Za-z0-9_]*)\s*>>")

#: Variables in the *generated* template, e.g. ``{{customer_name}}``.
OUTPUT_VARIABLE_RE = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")


class MissingPromptKeys(Exception):
    """The prompt template needs payload keys the request did not supply."""

    def __init__(self, keys: list[str]) -> None:
        self.keys = keys
        super().__init__(
            "payload is missing keys required by the prompt template: " + ", ".join(keys)
        )


def placeholders(template: str) -> set[str]:
    """Every payload key the template expects."""
    return {match.group(1) for match in _PLACEHOLDER_RE.finditer(template)}


def output_variables(text: str) -> set[str]:
    """Every ``{{variable}}`` referenced in generated template text."""
    return {match.group(1) for match in OUTPUT_VARIABLE_RE.finditer(text)}


def render(template: str, values: Mapping[str, Any]) -> str:
    """Substitute ``<<key>>`` markers from ``values``.

    Raises:
        MissingPromptKeys: if the template references a key that is absent or None.
    """
    missing = sorted(key for key in placeholders(template) if values.get(key) is None)
    if missing:
        raise MissingPromptKeys(missing)

    return _PLACEHOLDER_RE.sub(lambda m: _stringify(values[m.group(1)]), template)


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, indent=2, default=str)
    return str(value)
