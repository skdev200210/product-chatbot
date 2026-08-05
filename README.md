# Product Chatbot

FastAPI backend exposing a single chat endpoint backed by a [pydantic-ai](https://ai.pydantic.dev)
agent running on Claude. The agent's answer is schema-enforced, so the response
is always valid JSON.

## Layout

| File | Purpose |
| --- | --- |
| `app/main.py` | FastAPI app, `/chat` and `/health`, error mapping |
| `app/agent.py` | The pydantic-ai agent and one-turn orchestration |
| `app/system_prompt.py` | The system prompt template — **edit this** |
| `app/prompt.py` | Fills `{{placeholder}}` markers from the request payload |
| `app/schemas.py` | Request/response models, including the agent's output schema |
| `app/config.py` | Environment-driven settings |

## Setup

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt   # Linux/macOS: .venv/bin/pip
cp .env.example .env                            # then set ANTHROPIC_API_KEY
```

Run it:

```bash
.venv/Scripts/uvicorn app.main:app --reload
```

Interactive docs at http://127.0.0.1:8000/docs.

## The chat endpoint

`POST /chat` takes the whole conversation plus a free-form payload:

```json
{
  "conversation": [
    { "role": "user", "content": "Hi" },
    { "role": "assistant", "content": "Hello! How can I help?" },
    { "role": "user", "content": "What does Pro cost?" }
  ],
  "payload": {
    "product_name": "Acme Widgets",
    "product_context": "Widgets ship in 2 days. The Pro plan costs $49/month.",
    "customer_name": "Dana",
    "customer_plan": "Pro"
  }
}
```

The last message must have role `user` — it is the turn the agent answers.
Everything before it is replayed as conversation history.

Response:

```json
{
  "template": {
    "start_message": "Hi {{customer_name}}, this is Acme calling about your Pro plan.",
    "end_message": "Thanks for your time — have a good day.",
    "objective": "Confirm the customer wants to stay on Pro.",
    "instructions": "Be brief and warm. Let the customer finish speaking.",
    "rules": ["Never quote a price other than $49/month."],
    "summary_prompt": "Summarise the call and whether the customer renewed."
  },
  "created": null,
  "model": "claude-sonnet-5",
  "usage": { "input_tokens": 412, "output_tokens": 63 }
}
```

`template` is present on every turn. On the turn where the agent is actually
created, `created` holds the confirmation and `template` still carries the
template that was sent to the workflow API — so the client renders the same view
for both cases, and `created !== null` is the only check it needs:

```json
{
  "template": { "...": "the template the agent was created from" },
  "created": {
    "agent_name": "acme-renewals",
    "message": "Created the acme-renewals agent."
  },
  "model": "claude-sonnet-5",
  "usage": { "input_tokens": 980, "output_tokens": 120 }
}
```

## The prompt and payload injection

`SYSTEM_PROMPT_TEMPLATE` in `app/system_prompt.py` is the prompt. Any key written
in double curly braces is replaced at request time with the matching value from
`payload`:

```
You are the product assistant for {{product_name}}.
```

- Payload keys the prompt doesn't reference are ignored — send whatever you like.
- If the prompt references a key the payload omits (or sets to `null`), the
  request is rejected with `422` and the response names the missing keys.
- Values that aren't strings are stringified; dicts and lists become pretty JSON.
- Double braces are used rather than `str.format`, so single `{`/`}` (JSON
  examples, code snippets) and `$` (prices) need no escaping.

Change the placeholders by editing the template — no other code changes needed.

## Why the response is always JSON

The agent is created with `output_type=ChatReply`, so pydantic-ai constrains the
model to that schema and validates the result before returning it. A response
that doesn't fit is retried (`retries=2`) and, if it still fails, surfaces as a
`502` rather than as malformed output. To change the shape of the answer, edit
`ChatReply` in `app/schemas.py` — the JSON schema sent to the model, the
validation, and the OpenAPI docs all follow from it.

## Errors

| Status | Cause |
| --- | --- |
| `422` | Malformed body, last message not from the user, conversation too long, or a payload key the prompt needs is missing |
| `502` | The model call failed, or its output didn't satisfy the schema after retries |
| `504` | The model didn't respond within `REQUEST_TIMEOUT_SECONDS` |

## Configuration

All settings come from the environment (or `.env`); see `.env.example`.
`ANTHROPIC_API_KEY` is required and the app fails fast at import without it.

## Tests

```bash
.venv/Scripts/pip install -r requirements-dev.txt
.venv/Scripts/python -m pytest
```

Tests swap the real model for pydantic-ai's `TestModel` via `chat_agent.override(...)`,
so they need no API key and make no network calls.
