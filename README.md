# BrainstormLM

Open-source brainstorming model with a swappable generator and cached LLM clients.

See [`docs/brainstorm_pipeline_architecture.md`](docs/brainstorm_pipeline_architecture.md) for the
pipeline architecture and [`docs/steering_lineage.md`](docs/steering_lineage.md) for which steering
vectors ship.

## Quickstart

```
uv sync
cp .env.example .env       # then fill OPENAI_API_KEY / TOGETHER_API_KEY
uv run brainstorm serve    # start the FastAPI server (POST /brainstorm)
```
