# Project Conventions

## TDD by default
For new features and non-trivial bug fixes: Red → Green → Refactor.
1. Write a failing test first, in its own step.
2. Run it; confirm it fails for the right reason.
3. Minimum code to pass.
4. Refactor with tests green.

Skip TDD for spikes, config changes, or typo fixes — but say so explicitly
rather than silently writing implementation first.

When adding tests to existing code: write tests for the behavior assuming
the function doesn't exist, then verify the implementation passes.

## Git
- Imperative-mood commit messages ("add", not "added").
- One logical change per commit; tests ship with the code they cover.
- Branch naming: `type/short-description`.

## External services
Always use the CLI (`gh`, `aws`, `gcloud`, `kubectl`, `psql`) — never web UIs
or ad-hoc scripts.

## Verification
If a change can't be verified by a test, script, or screenshot, flag it
rather than claiming success.

## Commands
- Run all tests: `make test` (root suite, `uv run pytest --cov=src --cov-fail-under=35`). The two
  sub-packages have their own suites: `cd creativity_steering && python -m pytest` and
  `cd creativity_evaluators && python -m pytest`.
- Run one test: `uv run pytest tests/test_<name>.py::test_<fn> -q`
- Lint: `make lint` (`uv run ruff check .`); autofix with `uv run ruff check --fix .`
- Full local check: `make check` (lint + test)