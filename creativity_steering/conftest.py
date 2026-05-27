"""Make the standalone ``creativity_steering`` package importable.

This package uses a flat layout: the importable package lives at
``creativity_steering/creativity_steering``. When pytest is invoked from the repo root
(e.g. the root ``make test`` hook) it uses the root pyproject config, which does not know
about this package — so we prepend this folder to ``sys.path``. When pytest runs from inside
``creativity_steering/`` the ``pythonpath = ["."]`` setting in its own pyproject already
covers this and the insert below is a harmless no-op.
"""

import sys
from pathlib import Path

_PKG_ROOT = Path(__file__).parent  # holds the flat ``creativity_steering`` package
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

# The evaluator-stack filter (derivation/evaluator_filter.py) imports the sibling
# ``creativity_evaluators`` package, which lives at the repo root under a ``src`` layout
# (``creativity_evaluators/src/creativity_evaluators``). Put that ``src`` dir on the path so
# this package's tests can import it when run standalone (not just in a full-repo pytest run).
_CE_SRC = Path(__file__).parent.parent / "creativity_evaluators" / "src"
if (_CE_SRC / "creativity_evaluators" / "__init__.py").exists() and str(_CE_SRC) not in sys.path:
    sys.path.insert(0, str(_CE_SRC))
