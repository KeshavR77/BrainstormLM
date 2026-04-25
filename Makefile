check: lint test
lint:
	uv run ruff check .
test:
	uv run pytest --cov=src --cov-fail-under=35