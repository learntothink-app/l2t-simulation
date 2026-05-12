"""Tables, plots, HTML/LaTeX reports."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import asdict
from html import escape
from pathlib import Path
from typing import Iterable

import numpy as np

from .analysis import AggregateRow
from .invariants import InvariantReport
from .metrics import TrajectoryMetrics


log = logging.getLogger(__name__)

# Stable column order for the headline table.
METRIC_ORDER: tuple[str, ...] = (
    "m_mastery",
    "m_transfer",
    "m_retention_1d",
    "m_retention_3d",
    "m_retention_7d",
    "m_retention_14d",
    "m_hint",
    "m_robust",
    "m_efficiency",
    "m_meta",
    "m_engage",
    "m_calib",
)
POLICY_ORDER: tuple[str, ...] = ("B1_random", "B2_bkt", "B3_l2t_no_q", "B4_full_l2t")


# ---------------------------------------------------------------------------


def write_csv_table(rows: Iterable[AggregateRow], path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("methodology,policy,metric,mean,ci_low,ci_high,n\n")
        for r in rows:
            fh.write(
                f"{r.methodology},{r.policy},{r.metric},"
                f"{r.mean:.6f},{r.ci_low:.6f},{r.ci_high:.6f},{r.n}\n"
            )


def _fmt(mean: float, lo: float, hi: float) -> str:
    if any(math.isnan(x) for x in (mean, lo, hi)):
        return "—"
    return f"{mean:.3f} [{lo:.3f}, {hi:.3f}]"


def render_markdown_table(rows: list[AggregateRow], methodology: str) -> str:
    """Headline 4×12 table (metrics × policies) for one methodology."""
    by_pm: dict[tuple[str, str], AggregateRow] = {(r.policy, r.metric): r for r in rows if r.methodology == methodology}

    header = "| Metric | " + " | ".join(POLICY_ORDER) + " |\n"
    sep = "|---|" + "|".join(["---"] * len(POLICY_ORDER)) + "|\n"
    body = ""
    for met in METRIC_ORDER:
        cells = []
        for pol in POLICY_ORDER:
            r = by_pm.get((pol, met))
            if r is None:
                cells.append("—")
            else:
                cells.append(_fmt(r.mean, r.ci_low, r.ci_high))
        body += f"| {met} | " + " | ".join(cells) + " |\n"
    return header + sep + body


def render_latex_table(rows: list[AggregateRow], methodology: str) -> str:
    by_pm = {(r.policy, r.metric): r for r in rows if r.methodology == methodology}
    s = "\\begin{table}\n\\centering\n\\begin{tabular}{l" + "c" * len(POLICY_ORDER) + "}\n\\hline\n"
    s += "Metric & " + " & ".join(p.replace("_", "\\_") for p in POLICY_ORDER) + " \\\\\n\\hline\n"
    for met in METRIC_ORDER:
        cells = []
        for pol in POLICY_ORDER:
            r = by_pm.get((pol, met))
            cells.append(_fmt(r.mean, r.ci_low, r.ci_high) if r is not None else "---")
        s += met.replace("_", "\\_") + " & " + " & ".join(cells) + " \\\\\n"
    s += "\\hline\n\\end{tabular}\n"
    s += "\\caption{Headline results on the " + methodology + " methodology (mean and 95\\% bootstrap CI over $N=1000$ synthetic students).}\n"
    s += "\\label{tab:headline-" + methodology + "}\n\\end{table}\n"
    return s


# ---------------------------------------------------------------------------
# Matplotlib plots.


def _safe_import_mpl():
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:  # pragma: no cover
        log.warning("matplotlib unavailable: %s", exc)
        return None


def make_bar_charts(rows: list[AggregateRow], output_dir: str | os.PathLike) -> list[str]:
    plt = _safe_import_mpl()
    if plt is None:
        return []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []
    by_metric: dict[tuple[str, str], list[AggregateRow]] = {}
    for r in rows:
        by_metric.setdefault((r.methodology, r.metric), []).append(r)
    for (meth, metric), batch in by_metric.items():
        batch_sorted = sorted(batch, key=lambda r: POLICY_ORDER.index(r.policy) if r.policy in POLICY_ORDER else 99)
        names = [r.policy for r in batch_sorted]
        means = [r.mean for r in batch_sorted]
        errs_lo = [r.mean - r.ci_low for r in batch_sorted]
        errs_hi = [r.ci_high - r.mean for r in batch_sorted]
        fig, ax = plt.subplots(figsize=(6.0, 3.5))
        ax.bar(names, means, yerr=[errs_lo, errs_hi], capsize=4)
        ax.set_title(f"{metric} — {meth}")
        ax.set_ylabel(metric)
        ax.tick_params(axis="x", rotation=15)
        fig.tight_layout()
        png = output_dir / f"{meth}_{metric}.png"
        fig.savefig(png, dpi=150)
        plt.close(fig)
        saved.append(str(png))
    return saved


def make_robust_box(
    records_by_policy: dict[str, list[TrajectoryMetrics]],
    output_dir: str | os.PathLike,
    methodology: str,
) -> str | None:
    plt = _safe_import_mpl()
    if plt is None:
        return None
    fig, ax = plt.subplots(figsize=(6.0, 3.5))
    labels = []
    data = []
    for pol in POLICY_ORDER:
        recs = records_by_policy.get(pol, [])
        vals = [r.m_robust for r in recs if r.behaviour in {"guesser", "copier"} and not (isinstance(r.m_robust, float) and math.isnan(r.m_robust))]
        labels.append(pol)
        data.append(vals if vals else [0.0])
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_title(f"m_robust on guesser+copier — {methodology}")
    ax.set_ylabel("m_robust")
    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p = output_dir / f"{methodology}_m_robust_box.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    return str(p)


# ---------------------------------------------------------------------------
# HTML report.


def render_html_report(
    rows: list[AggregateRow],
    h1: dict,
    h2: dict,
    h3: dict,
    h4: dict,
    invariant_report: InvariantReport,
    methodologies: Iterable[str],
    figures: list[str],
    manifest: dict,
    output_path: str | os.PathLike,
    h1b: dict | None = None,
) -> None:
    parts: list[str] = []
    parts.append("<!doctype html><html><head><meta charset='utf-8'><title>L2T simulation report</title>")
    parts.append("<style>body{font-family:system-ui,sans-serif;max-width:980px;margin:2rem auto;padding:0 1rem;line-height:1.45}"
                 "table{border-collapse:collapse;margin:1rem 0}"
                 "th,td{border:1px solid #ccc;padding:.25rem .5rem;font-size:.9rem}"
                 "code{background:#f4f4f4;padding:.05rem .25rem;border-radius:.2rem}"
                 "h2{margin-top:2rem}</style></head><body>")
    parts.append("<h1>L2T simulation report</h1>")
    parts.append("<h2>Manifest</h2>")
    parts.append("<pre>" + escape(_pretty_manifest(manifest)) + "</pre>")
    for meth in methodologies:
        parts.append(f"<h2>Headline table — {escape(meth)}</h2>")
        parts.append(render_markdown_table([r for r in rows if r.methodology == meth], meth).replace("\n", "<br/>\n"))
    parts.append("<h2>Hypothesis tests</h2>")
    test_blocks = [("H1", h1)]
    if h1b is not None:
        test_blocks.append(("H1b", h1b))
    test_blocks.extend([("H2", h2), ("H3", h3), ("H4", h4)])
    for name, body in test_blocks:
        parts.append(f"<h3>{name}</h3><pre>" + escape(repr(body)) + "</pre>")
    parts.append("<h2>Model-checking the controller FSM</h2>")
    parts.append("<pre>" + escape(repr(invariant_report)) + "</pre>")
    if figures:
        parts.append("<h2>Figures</h2>")
        for f in figures:
            parts.append(f"<div><img src='{escape(os.path.relpath(f, Path(output_path).parent))}' /></div>")
    parts.append("</body></html>")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(parts), encoding="utf-8")


def _pretty_manifest(manifest: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in manifest.items())


# ---------------------------------------------------------------------------
# LaTeX fragment.


def render_latex_section(
    rows: list[AggregateRow],
    h1: dict,
    h2: dict,
    h3: dict,
    h4: dict,
    invariant_report: InvariantReport,
    methodologies: Iterable[str],
    output_path: str | os.PathLike,
    n_students: int,
    t_steps: int,
    h1b: dict | None = None,
) -> None:
    methodologies = list(methodologies)
    lines: list[str] = []
    lines.append("\\section{Preliminary Empirical Validation}\\label{sec:empirical}\n")
    lines.append("\\subsection{Simulation setup}\n")
    lines.append(
        f"We executed the simulation protocol of Sec.~\\ref{{sec:simulation}} on "
        f"$N={n_students}$ synthetic students with $T={t_steps}$ tutoring steps each, "
        "on two methodologies: (i)~a canonical introductory probability methodology "
        "covering sample spaces through conditional probability "
        "($10$ concepts, $12$ skills, $25$ training tasks, $18$ transfer tasks) and "
        "(ii)~a randomly-generated synthetic methodology used as a control for "
        "graph-structure effects ($K=10$ skills, $M=3$ meta-skills, $40$ tasks). "
        "The four behaviour types from Sec.~\\ref{sec:simulation}—honest, guesser, "
        "copier, help\\_seeker—appear in the proportions $0.60/0.15/0.10/0.15$.\n"
    )
    for meth in methodologies:
        lines.append(f"\\subsection{{Headline results: {meth}}}\n")
        lines.append(render_latex_table(rows, meth))
    lines.append("\\subsection{Hypothesis tests}\n")
    lines.append("\\paragraph{H1 (B4 > B3 on $m_{\\mathrm{robust}}$, guesser+copier, ablation).}\n")
    lines.append("\\verb|" + repr(h1) + "|\\par\n")
    if h1b is not None:
        lines.append("\\paragraph{H1b (L2T (B3$\\cup$B4) > B1 random on $m_{\\mathrm{robust}}$, pooled methodologies).}\n")
        lines.append("\\verb|" + repr(h1b) + "|\\par\n")
    lines.append("\\paragraph{H2 (B3,B4 > B2 on $m_{\\mathrm{transfer}}$ and $m_{\\mathrm{retention}}(7\\mathrm{d})$).}\n")
    lines.append("\\verb|" + repr(h2) + "|\\par\n")
    lines.append("\\paragraph{H3 (B2,B3,B4 > B1).}\n")
    lines.append("\\verb|" + repr(h3) + "|\\par\n")
    lines.append("\\paragraph{H4 (No L2T trajectory violates the three invariants).}\n")
    lines.append("\\verb|" + repr(h4) + "|\\par\n")
    lines.append("\\subsection{Model-checking the controller FSM}\n")
    lines.append(
        f"BFS exploration of the controller automaton ($|Q|={invariant_report.n_states}$, "
        f"$|\\delta|={invariant_report.n_transitions}$, "
        f"{invariant_report.n_traces_checked}~traces, "
        f"{invariant_report.seconds*1000:.1f}~ms) yields: "
        f"\\verb|theory_first={invariant_report.theory_first}, "
        f"no_spoiler={invariant_report.no_spoiler}, "
        f"transfer_gate={invariant_report.transfer_gate}|.\n"
    )
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")
