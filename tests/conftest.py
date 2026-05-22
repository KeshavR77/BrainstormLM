"""Test-time environment stubs.

Sets a placeholder OPENAI_API_KEY so ``Settings()`` can be constructed
without a real secret. ``TOGETHER_API_KEY`` is now Optional and doesn't
need a stub.
"""

from __future__ import annotations

import os

os.environ.setdefault("OPENAI_API_KEY", "test-stub-openai-key")
