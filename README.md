# BrainstormLM

Open-source brainstorming model with a swappable generator and cached LLM clients.

See [`docs/plans/plan-01-foundation.md`](docs/plans/plan-01-foundation.md) for the Phase 1 spec.

## Quickstart

```
uv sync
cp .env.example .env  # then fill OPENAI_API_KEY and TOGETHER_API_KEY
uv run brainstorm ask "name three unusual uses for a stapler"
```
