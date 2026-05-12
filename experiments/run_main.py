"""Top-level run: 4 policies × 2 methodologies × N students, full reporting.

Usage::

    python experiments/run_main.py --config experiments/configs/mini.yaml
    python experiments/run_main.py --config experiments/configs/medium.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running from project root without `pip install -e .`.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml  # type: ignore

from l2t_sim.analysis import aggregate_table, test_h1, test_h2, test_h3, test_h4
from l2t_sim.invariants import verify_invariants_on_fsm
from l2t_sim.methodology import (
    generate_synthetic_methodology,
    load_fano_methodology,
    save_synthetic_to_json,
)
from l2t_sim.reporting import (
    make_bar_charts,
    make_robust_box,
    render_html_report,
    render_latex_section,
    write_csv_table,
)
from l2t_sim.simulation import run_simulation
from l2t_sim.utils import collect_environment, save_json, setup_logging


log = logging.getLogger("run_main")

POLICY_NAMES = ("B1_random", "B2_bkt", "B3_l2t_no_q", "B4_full_l2t")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "experiments" / "configs" / "mini.yaml"),
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(ROOT / "results"),
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    results_root = Path(args.results_dir)
    log_dir = results_root / "raw"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(args.log_level, log_dir / f"run_{timestamp}.log")

    log.info("loaded config %s", cfg_path)
    log.info("config: %s", cfg)

    # Methodologies ---------------------------------------------------------
    fano_path = ROOT / "methodologies" / "fano.json"
    log.info("loading Fano methodology from %s", fano_path)
    fano = load_fano_methodology(fano_path)
    log.info(
        "fano: %d vertices, %d edges, %d skills, %d tasks, %d transfer",
        len(fano.hypergraph.vertices),
        len(fano.hypergraph.edges),
        len(fano.hypergraph.skill_ids),
        len(fano.tasks),
        len(fano.transfer_tasks),
    )

    log.info("generating synthetic methodology (seed=%d)", cfg["method_seed"])
    synth = generate_synthetic_methodology(seed=cfg["method_seed"])
    save_synthetic_to_json(synth, ROOT / "methodologies" / "synthetic_abstract.json")
    log.info(
        "synthetic: %d vertices, %d edges, %d skills, %d tasks",
        len(synth.hypergraph.vertices),
        len(synth.hypergraph.edges),
        len(synth.hypergraph.skill_ids),
        len(synth.tasks),
    )

    methodologies = {"fano": fano, "synthetic": synth}

    # Manifest --------------------------------------------------------------
    manifest = {
        "timestamp": timestamp,
        "config_path": str(cfg_path),
        "config": cfg,
        "environment": collect_environment(),
    }
    save_json(results_root / "experiment_manifest.json", manifest)

    # Run simulation --------------------------------------------------------
    all_results = {}
    for meth_name, meth in methodologies.items():
        for pol_name in POLICY_NAMES:
            t0 = time.perf_counter()
            log.info("running %s on %s", pol_name, meth_name)
            recs = run_simulation(
                policy_name=pol_name,
                methodology=meth,
                n_students=int(cfg["n_students"]),
                t_steps=int(cfg["t_steps"]),
                master_seed=int(cfg["master_seed"]),
                behaviour_proportions=cfg.get("behaviour_proportions"),
                retention_days=tuple(cfg.get("retention_intervals_days", [1, 3, 7, 14])),
                prereq_perturbation_eps=float(cfg.get("prereq_perturbation_eps", 0.10)),
                n_jobs=int(cfg.get("n_jobs", 1)),
            )
            secs = time.perf_counter() - t0
            log.info(
                "  done in %.1fs (%d records, mean m_mastery=%.3f, mean m_transfer=%.3f)",
                secs,
                len(recs),
                _safe_mean([r.m_mastery for r in recs]),
                _safe_mean([r.m_transfer for r in recs]),
            )
            all_results[(meth_name, pol_name)] = recs

    # Aggregate -------------------------------------------------------------
    rows = aggregate_table(all_results, n_bootstrap=int(cfg.get("n_bootstrap", 10000)))
    table_dir = results_root / "tables"
    write_csv_table(rows, table_dir / "main_results.csv")
    log.info("wrote %s", table_dir / "main_results.csv")

    # Hypotheses ------------------------------------------------------------
    fano_b3 = all_results[("fano", "B3_l2t_no_q")]
    fano_b4 = all_results[("fano", "B4_full_l2t")]
    fano_b2 = all_results[("fano", "B2_bkt")]
    fano_b1 = all_results[("fano", "B1_random")]
    h1 = test_h1(fano_b3, fano_b4)
    h2 = test_h2(fano_b2, fano_b3, fano_b4)
    h3 = test_h3(fano_b1, fano_b2, fano_b3, fano_b4)
    h4 = test_h4(fano_b3, fano_b4)
    log.info("H1: %s", h1)
    log.info("H4: %s", h4)
    save_json(table_dir / "hypothesis_tests.json", {"H1": h1, "H2": h2, "H3": h3, "H4": h4})

    # Model-checking --------------------------------------------------------
    log.info("running invariant verification (BFS on FSM)")
    invariant_report = verify_invariants_on_fsm(max_depth=int(cfg.get("fsm_bfs_depth", 30)))
    save_json(table_dir / "invariants_report.json", invariant_report.__dict__)

    # Figures ---------------------------------------------------------------
    fig_dir = results_root / "figures"
    figures = make_bar_charts(rows, fig_dir)
    for meth_name in methodologies:
        path = make_robust_box(
            {pol: all_results[(meth_name, pol)] for pol in POLICY_NAMES},
            fig_dir,
            meth_name,
        )
        if path:
            figures.append(path)

    # Reports ---------------------------------------------------------------
    render_html_report(
        rows=rows,
        h1=h1,
        h2=h2,
        h3=h3,
        h4=h4,
        invariant_report=invariant_report,
        methodologies=methodologies.keys(),
        figures=figures,
        manifest=manifest,
        output_path=results_root / "report.html",
    )
    render_latex_section(
        rows=rows,
        h1=h1,
        h2=h2,
        h3=h3,
        h4=h4,
        invariant_report=invariant_report,
        methodologies=methodologies.keys(),
        output_path=ROOT / "docs" / "paper_section_VIII.tex",
        n_students=int(cfg["n_students"]),
        t_steps=int(cfg["t_steps"]),
    )
    log.info("report at %s", results_root / "report.html")
    return 0


def _safe_mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and (x != x))]
    if not xs:
        return float("nan")
    return float(sum(xs) / len(xs))


if __name__ == "__main__":
    raise SystemExit(main())
