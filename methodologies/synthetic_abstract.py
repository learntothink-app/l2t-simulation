"""Generate and persist the synthetic abstract methodology used in §VII.

Usage:

    python -m methodologies.synthetic_abstract --seed 1 --output methodologies/synthetic_abstract.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# allow `python methodologies/synthetic_abstract.py` from project root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from l2t_sim.methodology import generate_synthetic_methodology, save_synthetic_to_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "methodologies" / "synthetic_abstract.json"),
    )
    args = parser.parse_args()

    md = generate_synthetic_methodology(seed=args.seed)
    save_synthetic_to_json(md, args.output)
    print(
        f"Wrote synthetic methodology to {args.output} "
        f"({len(md.hypergraph.vertices)} vertices, {len(md.hypergraph.edges)} edges)."
    )


if __name__ == "__main__":
    main()
