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
- Run all tests:
- Run one test:
- Lint:
- Full local check: