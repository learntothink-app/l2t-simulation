"""Stand-alone past-LTL invariant verification on the controller FSM.

Usage::

    python experiments/run_model_checking.py [--depth 30]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l2t_sim.invariants import verify_invariants_on_fsm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--depth", type=int, default=30, help="BFS depth bound")
    parser.add_argument("--j-star", type=int, default=4)
    args = parser.parse_args()

    rep = verify_invariants_on_fsm(max_depth=args.depth, j_star=args.j_star)
    print(json.dumps(rep.__dict__, indent=2))
    return 0 if rep.all_held else 1


if __name__ == "__main__":
    raise SystemExit(main())
