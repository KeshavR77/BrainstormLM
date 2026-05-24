"""Make the standalone ``creativity_steering`` package importable.

This package lives in its own folder with its own ``src/`` layout. When pytest is
invoked from the repo root (e.g. the root ``make test`` hook) it uses the root
pyproject config, which does not know about this package's source root — so we
prepend it here. When pytest is run from inside ``creativity_steering/`` the
``pythonpath = ["src"]`` setting in its own pyproject already covers this, and the
insert below is a harmless no-op.
"""

import sys
from pathlib import Path

_SRC = Path(__file__).parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# The evaluator-stack filter (derivation/evaluator_filter.py) imports the sibling
# ``creativity_evaluators`` package, which lives at the repo root. Put it on the path so
# this package's tests can import it when run standalone (not just in a full-repo pytest run).
_CE = Path(__file__).parent.parent / "creativity_evaluators"
if (_CE / "creativity_evaluators" / "__init__.py").exists() and str(_CE) not in sys.path:
    sys.path.insert(0, str(_CE))
