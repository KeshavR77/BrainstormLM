"""Make the standalone ``creativity_evaluators`` package importable.

This package uses a flat layout: the importable package lives at
``creativity_evaluators/creativity_evaluators``. When pytest is invoked from the repo
root (rather than from inside this folder) the root pyproject does not know about this
package, so we prepend this folder to ``sys.path``. When pytest runs from inside this
folder the ``pythonpath = ["."]`` setting in the local pyproject already covers it and
the insert below is a harmless no-op.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
