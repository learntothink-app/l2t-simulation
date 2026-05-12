"""End-to-end mini-run: all 4 policies on both methodologies, N=10, T=20.

This is the single most useful smoke test for the simulation pipeline; it
exercises every module and must finish in a few seconds.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from l2t_sim.methodology import generate_synthetic_methodology, load_fano_methodology
from l2t_sim.simulation import run_simulation


ROOT = Path(__file__).resolve().parent.parent
FANO = ROOT / "methodologies" / "fano.json"


@pytest.fixture(scope="module")
def methodologies():
    fano = load_fano_methodology(FANO) if FANO.exists() else None
    synth = generate_synthetic_methodology(seed=1)
    out = {}
    if fano is not None:
        out["fano"] = fano
    out["synthetic"] = synth
    return out


@pytest.mark.parametrize("policy", ["B1_random", "B2_bkt", "B3_l2t_no_q", "B4_full_l2t"])
def test_end_to_end_mini(methodologies, policy):
    for name, md in methodologies.items():
        recs = run_simulation(
            policy_name=policy,
            methodology=md,
            n_students=10,
            t_steps=20,
            master_seed=7,
            n_jobs=1,
        )
        assert len(recs) == 10
        for r in recs:
            assert r.n_steps == 20
            assert isinstance(r.m_transfer, float)
            assert not math.isnan(r.m_mastery)


def test_invariants_held_for_l2t(methodologies):
    md = next(iter(methodologies.values()))
    for policy in ("B3_l2t_no_q", "B4_full_l2t"):
        recs = run_simulation(
            policy_name=policy,
            methodology=md,
            n_students=10,
            t_steps=20,
            master_seed=11,
            n_jobs=1,
        )
        for r in recs:
            assert r.invariants_held["theory_first"]
            assert r.invariants_held["no_spoiler"]
            # transfer_gate need not hold within T=20 if block isn't completed
