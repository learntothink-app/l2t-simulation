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


def test_h3(
    b1: list[TrajectoryMetrics],
    b2: list[TrajectoryMetrics],
    b3: list[TrajectoryMetrics],
    b4: list[TrajectoryMetrics],
) -> dict:
    """H3 sanity: B2/B3/B4 outperform B1 on every key metric."""
    metrics = ("m_mastery", "m_transfer", "m_retention_7d", "m_efficiency")
    out: dict = {}
    b1_metrics = {attr: _values(b1, attr) for attr in metrics}
    for label, run in (("b2", b2), ("b3", b3), ("b4", b4)):
        for attr in metrics:
            arr = _values(run, attr)
            base = b1_metrics[attr]
            if arr.size == 0 or base.size == 0:
                continue
            t, p = _welch_t_one_sided(arr, base)
            out[f"{attr}_{label}_vs_b1"] = {
                "mean_b1": float(base.mean()),
                "mean_other": float(arr.mean()),
                "t": t,
                "p_value": p,
                "cohen_d": cohen_d(arr, base),
                "confirmed": bool(p < 0.05),
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
