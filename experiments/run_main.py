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

from l2t_sim.analysis import (
    aggregate_table,
    test_h1,
    test_h1b,
    test_h2,
    test_h3,
    test_h4,
    test_inference_robust_breakdown,
)
from l2t_sim.invariants import verify_invariants_on_fsm
from l2t_sim.methodology import (
    generate_synthetic_methodology,
    load_probability_methodology,
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
        help=(
            "Path to a config YAML, OR a short name (e.g. 'medium') that "
            "resolves to experiments/configs/<name>.yaml."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override master_seed from the config (paper §VIII.A uses --seed 42).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(ROOT / "results"),
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    # Resolve --config shortcut: "medium" → experiments/configs/medium.yaml.
    raw_cfg = args.config
    if "/" not in raw_cfg and not raw_cfg.endswith((".yaml", ".yml")):
        cfg_path = ROOT / "experiments" / "configs" / f"{raw_cfg}.yaml"
    else:
        cfg_path = Path(raw_cfg)

    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)

    if args.seed is not None:
        cfg["master_seed"] = int(args.seed)

    results_root = Path(args.results_dir)
    log_dir = results_root / "raw"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    setup_logging(args.log_level, log_dir / f"run_{timestamp}.log")

    log.info("loaded config %s", cfg_path)
    log.info("config: %s", cfg)

    # Methodologies ---------------------------------------------------------
    prob_path = ROOT / "methodologies" / "probability_basics.json"
    log.info("loading probability methodology from %s", prob_path)
    probability = load_probability_methodology(prob_path)
    log.info(
        "probability: %d vertices, %d edges, %d skills, %d tasks, %d transfer, %d holdout",
        len(probability.hypergraph.vertices),
        len(probability.hypergraph.edges),
        len(probability.hypergraph.skill_ids),
        len(probability.tasks),
        len(probability.transfer_tasks),
        len(probability.holdout_transfer_ids),
    )

    # v0.2.0: synthetic is an ensemble over K random graph topologies.
    # K=1 reproduces v0.1.x single-graph behaviour.
    K = int(cfg.get("synthetic_ensemble_k", 1))
    if K == 1:
        seed = int(cfg.get("method_seed", 1))
        log.info("generating synthetic methodology (seed=%d)", seed)
        synth = generate_synthetic_methodology(seed=seed)
        save_synthetic_to_json(synth, ROOT / "methodologies" / "synthetic_abstract.json")
        synthetic_ensemble = [(seed, synth)]
    else:
        log.info("synthetic ensemble: K=%d random graphs (seeds 1..%d)", K, K)
        synthetic_ensemble = []
        for graph_seed in range(1, K + 1):
            synth = generate_synthetic_methodology(seed=graph_seed)
            synthetic_ensemble.append((graph_seed, synth))
        # Save the seed=1 graph for inspection; the others are reproducible
        # from their seeds.
        save_synthetic_to_json(
            synthetic_ensemble[0][1], ROOT / "methodologies" / "synthetic_abstract.json"
        )
    log.info(
        "synthetic (seed=%d sample): %d vertices, %d edges, %d skills, %d tasks, %d transfer, %d holdout",
        synthetic_ensemble[0][0],
        len(synthetic_ensemble[0][1].hypergraph.vertices),
        len(synthetic_ensemble[0][1].hypergraph.edges),
        len(synthetic_ensemble[0][1].hypergraph.skill_ids),
        len(synthetic_ensemble[0][1].tasks),
        len(synthetic_ensemble[0][1].transfer_tasks),
        len(synthetic_ensemble[0][1].holdout_transfer_ids),
    )

    methodologies = {"probability": probability, "synthetic": synthetic_ensemble[0][1]}

    # Manifest --------------------------------------------------------------
    manifest = {
        "timestamp": timestamp,
        "config_path": str(cfg_path),
        "config": cfg,
        "environment": collect_environment(),
    }
    save_json(results_root / "experiment_manifest.json", manifest)

    # Run simulation --------------------------------------------------------
    all_results: dict[tuple[str, str], list] = {}
    # Per-graph synthetic results: {(graph_seed, pol_name): records}
    synthetic_per_graph: dict[tuple[int, str], list] = {}

    sim_kwargs = dict(
        n_students=int(cfg["n_students"]),
        t_steps=int(cfg["t_steps"]),
        master_seed=int(cfg["master_seed"]),
        behaviour_proportions=cfg.get("behaviour_proportions"),
        retention_days=tuple(cfg.get("retention_intervals_days", [1, 3, 7, 14])),
        prereq_perturbation_eps=float(cfg.get("prereq_perturbation_eps", 0.10)),
        n_jobs=int(cfg.get("n_jobs", 1)),
    )

    # Probability — single methodology run.
    for pol_name in POLICY_NAMES:
        t0 = time.perf_counter()
        log.info("running %s on probability", pol_name)
        recs = run_simulation(policy_name=pol_name, methodology=probability, **sim_kwargs)
        secs = time.perf_counter() - t0
        log.info(
            "  done in %.1fs (%d records, mean m_mastery=%.3f, mean m_transfer=%.3f)",
            secs,
            len(recs),
            _safe_mean([r.m_mastery for r in recs]),
            _safe_mean([r.m_transfer for r in recs]),
        )
        all_results[("probability", pol_name)] = recs

    # Synthetic — K-graph ensemble. Trajectories are pooled across graphs
    # under ('synthetic', pol_name); the per-graph means feed into
    # synthetic_aggregate.csv (mean ± std over graphs).
    synth_pooled: dict[str, list] = {p: [] for p in POLICY_NAMES}
    for graph_seed, synth_meth in synthetic_ensemble:
        for pol_name in POLICY_NAMES:
            t0 = time.perf_counter()
            log.info("running %s on synthetic graph seed=%d", pol_name, graph_seed)
            recs = run_simulation(policy_name=pol_name, methodology=synth_meth, **sim_kwargs)
            secs = time.perf_counter() - t0
            log.info(
                "  done in %.1fs (%d records, mean m_mastery=%.3f, mean m_transfer=%.3f)",
                secs,
                len(recs),
                _safe_mean([r.m_mastery for r in recs]),
                _safe_mean([r.m_transfer for r in recs]),
            )
            synthetic_per_graph[(graph_seed, pol_name)] = recs
            synth_pooled[pol_name].extend(recs)
    for pol_name in POLICY_NAMES:
        all_results[("synthetic", pol_name)] = synth_pooled[pol_name]

    # Aggregate -------------------------------------------------------------
    rows = aggregate_table(all_results, n_bootstrap=int(cfg.get("n_bootstrap", 10000)))
    table_dir = results_root / "tables"
    write_csv_table(rows, table_dir / "main_results.csv")
    log.info("wrote %s", table_dir / "main_results.csv")

    # Per-graph synthetic outputs (only meaningful when K > 1).
    if K > 1:
        _write_synthetic_per_graph_csv(
            synthetic_per_graph, table_dir / "synthetic_per_graph.csv"
        )
        _write_synthetic_aggregate_csv(
            synthetic_per_graph, table_dir / "synthetic_aggregate.csv"
        )
        log.info("wrote %s", table_dir / "synthetic_per_graph.csv")
        log.info("wrote %s", table_dir / "synthetic_aggregate.csv")

    # Hypotheses ------------------------------------------------------------
    # POOLED testing: combine all methodologies for the primary statistical
    # tests. Per-methodology breakdowns are saved separately for
    # transparency in §VIII.
    pooled: dict[str, list] = {p: [] for p in POLICY_NAMES}
    for (_meth, pol), recs in all_results.items():
        pooled[pol].extend(recs)

    h1 = test_h1(pooled["B3_l2t_no_q"], pooled["B4_full_l2t"])
    h1b = test_h1b(pooled["B1_random"], pooled["B3_l2t_no_q"], pooled["B4_full_l2t"])
    h2 = test_h2(pooled["B2_bkt"], pooled["B3_l2t_no_q"], pooled["B4_full_l2t"])
    h3 = test_h3(pooled["B1_random"], pooled["B2_bkt"], pooled["B3_l2t_no_q"], pooled["B4_full_l2t"])
    h4 = test_h4(pooled["B3_l2t_no_q"], pooled["B4_full_l2t"])

    per_methodology: dict[str, dict] = {"H2": {}, "H3": {}, "H1b": {}}
    for meth_name in methodologies:
        b1 = all_results[(meth_name, "B1_random")]
        b2 = all_results[(meth_name, "B2_bkt")]
        b3 = all_results[(meth_name, "B3_l2t_no_q")]
        b4 = all_results[(meth_name, "B4_full_l2t")]
        per_methodology["H2"][meth_name] = test_h2(b2, b3, b4)
        per_methodology["H3"][meth_name] = test_h3(b1, b2, b3, b4)
        per_methodology["H1b"][meth_name] = test_h1b(b1, b3, b4)

    log.info("H1: %s", h1)
    log.info("H1b: %s", h1b)
    log.info("H4: %s", h4)
    save_json(
        table_dir / "hypothesis_tests.json",
        {
            "H1": h1,
            "H1b": h1b,
            "H2": h2,
            "H3": h3,
            "H4": h4,
            "_per_methodology": per_methodology,
        },
    )

    # v0.2.0: dedicated m_inference_robust report — pooled + per
    # methodology + per behaviour subgroup. Sign convention: positive d
    # means the L2T policy beats Random on inference quality.
    inference_robust_report: dict = {
        "metric": "m_inference_robust",
        "description": (
            "Mean absolute error between final per-skill belief b_T and "
            "ground-truth p_true_T, restricted to contaminated students "
            "(behaviour in {guesser, copier}). Lower raw value = better; "
            "cohen_d is reported with positive sign meaning the policy "
            "beats Random."
        ),
        "pooled": {
            "B3_vs_B1": test_h3(
                pooled["B1_random"], pooled["B2_bkt"], pooled["B3_l2t_no_q"], pooled["B4_full_l2t"]
            )["m_inference_robust_b3_vs_b1"],
            "B4_vs_B1": test_h3(
                pooled["B1_random"], pooled["B2_bkt"], pooled["B3_l2t_no_q"], pooled["B4_full_l2t"]
            )["m_inference_robust_b4_vs_b1"],
            "per_behaviour": test_inference_robust_breakdown(
                pooled["B1_random"], pooled["B3_l2t_no_q"], pooled["B4_full_l2t"]
            ),
        },
        "per_methodology": {},
    }
    for meth_name in methodologies:
        b1 = all_results[(meth_name, "B1_random")]
        b3 = all_results[(meth_name, "B3_l2t_no_q")]
        b4 = all_results[(meth_name, "B4_full_l2t")]
        meth_h3 = test_h3(
            b1, all_results[(meth_name, "B2_bkt")], b3, b4
        )
        inference_robust_report["per_methodology"][meth_name] = {
            "B3_vs_B1": meth_h3.get("m_inference_robust_b3_vs_b1"),
            "B4_vs_B1": meth_h3.get("m_inference_robust_b4_vs_b1"),
            "per_behaviour": test_inference_robust_breakdown(b1, b3, b4),
        }
    save_json(table_dir / "inference_robust.json", inference_robust_report)
    log.info("wrote %s", table_dir / "inference_robust.json")

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
        h1b=h1b,
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
        h1b=h1b,
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


_PER_GRAPH_METRICS = (
    "m_mastery",
    "m_transfer",
    "m_retention_1d",
    "m_retention_3d",
    "m_retention_7d",
    "m_retention_14d",
    "m_hint",
    "m_robust",
    "m_inference_robust",
    "m_efficiency",
    "m_meta",
    "m_engage",
    "m_calib",
)


def _per_graph_means(per_graph: dict[tuple[int, str], list]) -> dict[tuple[int, str, str], tuple[float, int]]:
    """Per-(graph_seed, policy, metric) → (mean, n) with NaN filtered."""
    import math

    out: dict[tuple[int, str, str], tuple[float, int]] = {}
    for (graph_seed, pol_name), recs in per_graph.items():
        for metric in _PER_GRAPH_METRICS:
            values = [
                getattr(r, metric)
                for r in recs
                if getattr(r, metric) is not None
                and not (isinstance(getattr(r, metric), float) and math.isnan(getattr(r, metric)))
            ]
            if not values:
                continue
            out[(graph_seed, pol_name, metric)] = (float(sum(values) / len(values)), len(values))
    return out


def _write_synthetic_per_graph_csv(per_graph: dict[tuple[int, str], list], path: Path) -> None:
    """Long-format CSV: one row per (graph_seed, policy, metric)."""
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    means = _per_graph_means(per_graph)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["graph_seed", "policy", "metric", "mean", "n_students"])
        for (graph_seed, pol_name, metric), (mean, n) in sorted(means.items()):
            w.writerow([graph_seed, pol_name, metric, f"{mean:.6f}", n])


def _write_synthetic_aggregate_csv(per_graph: dict[tuple[int, str], list], path: Path) -> None:
    """Wide-format CSV: one row per (policy, metric) with mean and std over K graphs."""
    import csv
    import math
    from collections import defaultdict

    path.parent.mkdir(parents=True, exist_ok=True)
    means = _per_graph_means(per_graph)
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (graph_seed, pol_name, metric), (mean, _n) in means.items():
        grouped[(pol_name, metric)].append(mean)

    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "metric", "mean_over_graphs", "std_over_graphs", "k"])
        for (pol_name, metric), values in sorted(grouped.items()):
            k = len(values)
            mean = sum(values) / k
            if k > 1:
                var = sum((v - mean) ** 2 for v in values) / (k - 1)
                std = math.sqrt(var)
            else:
                std = 0.0
            w.writerow([pol_name, metric, f"{mean:.6f}", f"{std:.6f}", k])


if __name__ == "__main__":
    raise SystemExit(main())
