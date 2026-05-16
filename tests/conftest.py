"""Test-time environment stubs.

`brainstormlm.api.server` instantiates ``app = build_app()`` at module import,
which calls ``get_settings()`` -> ``Settings()`` (pydantic-settings) and requires
``OPENAI_API_KEY`` and ``TOGETHER_API_KEY``. Real values aren't needed for the
test suite (network calls are stubbed or skipped); the imports just have to
succeed. Set placeholder secrets here, before any test module is collected.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENAI_API_KEY", "test-stub-openai-key")
os.environ.setdefault("TOGETHER_API_KEY", "test-stub-together-key")
