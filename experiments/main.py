"""Paper-cited entry point — thin alias for ``experiments/run_main.py``.

The paper's §VIII.A reproducibility block is::

    python experiments/main.py --seed 42 --config medium

This file exists so that exact incantation works. All real work is in
``run_main.py``; this module simply delegates. It is kept tiny so that
the paper-cited command and the historical command stay in sync without
duplication.
"""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_main import main as _run_main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(_run_main())
