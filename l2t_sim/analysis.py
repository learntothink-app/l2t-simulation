"""Bootstrap CIs and H1–H4 hypothesis tests (§VII.C)."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from .metrics import TrajectoryMetrics


log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------


def bootstrap_ci(
    values: Sequence[float] | np.ndarray,
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """Empirical mean + percentile bootstrap CI."""

    arr = np.asarray([v for v in values if not (isinstance(v, float) and math.isnan(v))], dtype=float)
    if arr.size == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = arr.size
    idx = rng.integers(0, n, size=(n_resamples, n))
    boot_means = arr[idx].mean(axis=1)
    alpha = (1.0 - ci) / 2.0
    return (
        float(arr.mean()),
        float(np.quantile(boot_means, alpha)),
        float(np.quantile(boot_means, 1.0 - alpha)),
    )


def _welch_t_one_sided(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """One-sided Welch t-test of H0: mean(a) ≤ mean(b) vs. H1: mean(a) > mean(b).

    Returns (t-statistic, p-value).  Falls back to scipy if available.
    """
    try:
        from scipy.stats import ttest_ind

        res = ttest_ind(a, b, equal_var=False, alternative="greater")
        return float(res.statistic), float(res.pvalue)
    except Exception:  # pragma: no cover
        # plain Welch fallback
        ma, mb = a.mean(), b.mean()
        va, vb = a.var(ddof=1), b.var(ddof=1)
        na, nb = a.size, b.size
        denom = math.sqrt(va / na + vb / nb)
        if denom == 0:
            return 0.0, 0.5
        t = (ma - mb) / denom
        # crude p via standard normal
        p = 0.5 * math.erfc(t / math.sqrt(2))
        return float(t), float(p)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d for two independent samples, pooled SD."""
    if a.size < 2 or b.size < 2:
        return 0.0
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    pooled = math.sqrt(0.5 * (s1 ** 2 + s2 ** 2))
    if pooled == 0:
        return 0.0
    return float((a.mean() - b.mean()) / pooled)


# ---------------------------------------------------------------------------
# Hypothesis tests.


def _values(records: Iterable[TrajectoryMetrics], attr: str, beh_filter: set | None = None) -> np.ndarray:
    out = []
    for r in records:
        if beh_filter is not None and r.behaviour not in beh_filter:
            continue
        v = getattr(r, attr)
        if v is None:
            continue
        if isinstance(v, float) and math.isnan(v):
            continue
        out.append(float(v))
    return np.asarray(out, dtype=float)


def test_h1(b3: list[TrajectoryMetrics], b4: list[TrajectoryMetrics]) -> dict:
    """H1: B4 > B3 on m_robust over the (guesser, copier) subsample."""
    filt = {"guesser", "copier"}
    b3_v = _values(b3, "m_robust", filt)
    b4_v = _values(b4, "m_robust", filt)
    if b3_v.size == 0 or b4_v.size == 0:
        return {
            "metric": "m_robust",
            "subsample": "guesser+copier",
            "n_b3": int(b3_v.size),
            "n_b4": int(b4_v.size),
            "t": None,
            "p_value": None,
            "cohen_d": None,
            "confirmed": False,
            "note": "insufficient low-q transfer observations to evaluate",
        }
    t, p = _welch_t_one_sided(b4_v, b3_v)
    d = cohen_d(b4_v, b3_v)
    return {
        "metric": "m_robust",
        "subsample": "guesser+copier",
        "n_b3": int(b3_v.size),
        "n_b4": int(b4_v.size),
        "mean_b3": float(b3_v.mean()),
        "mean_b4": float(b4_v.mean()),
        "t": float(t),
        "p_value": float(p),
        "cohen_d": float(d),
        "confirmed": bool(p < 0.05 and d > 0.2),
    }


def test_h1b(
    b1: list[TrajectoryMetrics],
    b3: list[TrajectoryMetrics],
    b4: list[TrajectoryMetrics],
) -> dict:
    """H1b: pooled L2T (B3 ∪ B4) > Random (B1) on m_robust.

    H1 (test_h1) tests the ablation B4>B3 — whether reliability
    weighting *adds anything* on top of rule-based L2T. H1b tests the
    main pedagogical claim: that L2T's curriculum yields more robust
    correctness on contaminated (low-q) transfer attempts than Random.

    Both methodologies are pooled — m_robust is per-student and has
    NaN whenever the student had no low-q transfer attempts; those
    are filtered out by ``_values``.
    """

    b1_v = _values(b1, "m_robust")
    b34_v = _values(list(b3) + list(b4), "m_robust")
    if b1_v.size == 0 or b34_v.size == 0:
        return {
            "metric": "m_robust",
            "comparison": "L2T (B3+B4) vs Random (B1), pooled methodologies",
            "n_b1": int(b1_v.size),
            "n_l2t": int(b34_v.size),
            "t": None,
            "p_value": None,
            "cohen_d": None,
            "confirmed": False,
            "note": "insufficient low-q transfer observations to evaluate",
        }
    t, p = _welch_t_one_sided(b34_v, b1_v)
    d = cohen_d(b34_v, b1_v)
    return {
        "metric": "m_robust",
        "comparison": "L2T (B3+B4) vs Random (B1), pooled methodologies",
        "n_b1": int(b1_v.size),
        "n_l2t": int(b34_v.size),
        "mean_b1": float(b1_v.mean()),
        "mean_l2t": float(b34_v.mean()),
        "t": float(t),
        "p_value": float(p),
        "cohen_d": float(d),
        "confirmed": bool(p < 0.05 and d > 0.2),
    }


def test_h2(
    b2: list[TrajectoryMetrics],
    b3: list[TrajectoryMetrics],
    b4: list[TrajectoryMetrics],
) -> dict:
    """H2: B3, B4 > B2 on m_transfer and m_retention(7d)."""
    results: dict = {}
    for attr in ("m_transfer", "m_retention_7d"):
        b2_v = _values(b2, attr)
        for label, arr in (("b3", _values(b3, attr)), ("b4", _values(b4, attr))):
            t, p = _welch_t_one_sided(arr, b2_v) if arr.size and b2_v.size else (None, None)
            d = cohen_d(arr, b2_v) if arr.size and b2_v.size else None
            results[f"{attr}_{label}_vs_b2"] = {
                "n_b2": int(b2_v.size),
                "n_other": int(arr.size),
                "mean_b2": float(b2_v.mean()) if b2_v.size else None,
                "mean_other": float(arr.mean()) if arr.size else None,
                "t": t,
                "p_value": p,
                "cohen_d": d,
                "confirmed": bool(p is not None and p < 0.05 and (d or 0) > 0.2),
            }
    return results


_H3_LOWER_IS_BETTER = frozenset({"m_inference_robust"})


def test_h3(
    b1: list[TrajectoryMetrics],
    b2: list[TrajectoryMetrics],
    b3: list[TrajectoryMetrics],
    b4: list[TrajectoryMetrics],
) -> dict:
    """H3 sanity: B2/B3/B4 outperform B1 on every key metric.

    For metrics where lower-is-better (currently only m_inference_robust)
    the one-sided test flips: H1 becomes mean(arr) < mean(base). Cohen's
    d is reported with sign such that **positive d = better than B1**
    in all cells, so the same threshold ``d > 0.2`` works uniformly.
    """

    metrics = (
        "m_mastery",
        "m_transfer",
        "m_retention_7d",
        "m_efficiency",
        "m_inference_robust",
    )
    out: dict = {}
    b1_metrics = {attr: _values(b1, attr) for attr in metrics}
    for label, run in (("b2", b2), ("b3", b3), ("b4", b4)):
        for attr in metrics:
            arr = _values(run, attr)
            base = b1_metrics[attr]
            if arr.size == 0 or base.size == 0:
                continue
            lower_is_better = attr in _H3_LOWER_IS_BETTER
            if lower_is_better:
                # Test base > arr (i.e. B1 has higher error than other policy).
                t, p = _welch_t_one_sided(base, arr)
                # Flip d so that positive => other policy better than B1.
                d = -cohen_d(arr, base)
            else:
                t, p = _welch_t_one_sided(arr, base)
                d = cohen_d(arr, base)
            out[f"{attr}_{label}_vs_b1"] = {
                "mean_b1": float(base.mean()),
                "mean_other": float(arr.mean()),
                "t": t,
                "p_value": p,
                "cohen_d": d,
                "lower_is_better": lower_is_better,
                "n_b1": int(base.size),
                "n_other": int(arr.size),
                "confirmed": bool(p < 0.05 and d > 0.2),
            }
    return out


def test_inference_robust_b4_vs_b3(
    b3: list[TrajectoryMetrics],
    b4: list[TrajectoryMetrics],
) -> dict:
    """B4 vs B3 ablation on m_inference_robust — the v0.2.0 'new H1'.

    Tests whether the q_t reliability weighting (B4) adds isolated
    benefit over rule-based L2T (B3, q_t=1) on belief quality for
    contaminated students. Independent of the challenge-battery test
    (Task 3) which is a triangulation on a fixed observation stream.

    Sign convention: positive ``cohen_d`` means B4 beats B3 (i.e. B4
    has lower MAE between belief and truth).
    """

    b3_v = _values(b3, "m_inference_robust")
    b4_v = _values(b4, "m_inference_robust")
    if b3_v.size == 0 or b4_v.size == 0:
        return {
            "metric": "m_inference_robust",
            "comparison": "B4 vs B3 (q_t isolated ablation)",
            "n_b3": int(b3_v.size),
            "n_b4": int(b4_v.size),
            "note": "insufficient inference_robust observations (need contaminated students)",
            "confirmed": False,
        }
    # Lower is better. Test base=b3 > arr=b4 ⇒ B4 better (mean_b4 < mean_b3).
    t, p = _welch_t_one_sided(b3_v, b4_v)
    d = -cohen_d(b4_v, b3_v)
    return {
        "metric": "m_inference_robust",
        "comparison": "B4 vs B3 (q_t isolated ablation, positive d = B4 better)",
        "n_b3": int(b3_v.size),
        "n_b4": int(b4_v.size),
        "mean_b3": float(b3_v.mean()),
        "mean_b4": float(b4_v.mean()),
        "t": float(t),
        "p_value": float(p),
        "cohen_d": float(d),
        "confirmed": bool(p < 0.05 and d > 0.2),
    }


def test_inference_robust_breakdown(
    b1: list[TrajectoryMetrics],
    b3: list[TrajectoryMetrics],
    b4: list[TrajectoryMetrics],
) -> dict:
    """Per-behaviour breakdown of m_inference_robust for B3, B4 vs B1.

    Returns a dict keyed by ``"{policy}_vs_b1_{behaviour}"`` with
    means, n, cohen_d (positive = policy better than B1), p_value, CI.
    Probabilities are not bootstrapped here — that lives in
    aggregate_table; this function is for the dedicated inference table.
    """

    out: dict = {}
    for policy_label, run in (("b3", b3), ("b4", b4)):
        for behaviour in ("guesser", "copier"):
            filt = {behaviour}
            base = _values(b1, "m_inference_robust", filt)
            arr = _values(run, "m_inference_robust", filt)
            if base.size == 0 or arr.size == 0:
                out[f"{policy_label}_vs_b1_{behaviour}"] = {
                    "n_b1": int(base.size),
                    "n_other": int(arr.size),
                    "note": "insufficient observations",
                }
                continue
            t, p = _welch_t_one_sided(base, arr)  # base > arr → policy better
            d = -cohen_d(arr, base)
            out[f"{policy_label}_vs_b1_{behaviour}"] = {
                "behaviour": behaviour,
                "n_b1": int(base.size),
                "n_other": int(arr.size),
                "mean_b1": float(base.mean()),
                "mean_other": float(arr.mean()),
                "t": float(t),
                "p_value": float(p),
                "cohen_d": float(d),
                "confirmed": bool(p < 0.05 and d > 0.2),
            }
    return out


def test_h4(
    b3: list[TrajectoryMetrics],
    b4: list[TrajectoryMetrics],
) -> dict:
    """H4: no L2T trajectory violates the three invariants."""
    def violations(rs: list[TrajectoryMetrics]) -> int:
        return sum(1 for r in rs if not all(r.invariants_held.values()))

    v3 = violations(b3)
    v4 = violations(b4)
    return {
        "violations_b3": v3,
        "violations_b4": v4,
        "n_b3": len(b3),
        "n_b4": len(b4),
        "h4_confirmed": v3 == 0 and v4 == 0,
    }


# ---------------------------------------------------------------------------


@dataclass
class AggregateRow:
    methodology: str
    policy: str
    metric: str
    mean: float
    ci_low: float
    ci_high: float
    n: int


def aggregate_table(
    results_by_pair: dict[tuple[str, str], list[TrajectoryMetrics]],
    n_bootstrap: int = 10_000,
    seed: int = 0,
) -> list[AggregateRow]:
    rows: list[AggregateRow] = []
    metrics = (
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
    for (meth, pol), recs in results_by_pair.items():
        for attr in metrics:
            vals = _values(recs, attr)
            mean, lo, hi = bootstrap_ci(vals, n_resamples=n_bootstrap, seed=seed)
            rows.append(AggregateRow(meth, pol, attr, mean, lo, hi, int(vals.size)))
    return rows
