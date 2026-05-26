"""Make the standalone ``creativity_evaluators`` package importable.

This package uses a ``src`` layout: the importable package lives at
``creativity_evaluators/src/creativity_evaluators``. When pytest is invoked from the repo
root (rather than from inside this folder) the root pyproject does not know about this
package, so we prepend the local ``src`` folder to ``sys.path``. When pytest runs from
inside this folder the ``pythonpath = ["src"]`` setting in the local pyproject already
covers it and the insert below is a harmless no-op.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
