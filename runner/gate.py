"""
runner/gate.py  —  THE JUDGE. The loop must NOT edit this file.

The promotion gate. Stops the loop from pushing NOISE as if it were improvement.
Four stacked checks, cheapest first:

  1. noise floor   — sigma of the fixed baseline, measured ONCE. Now consumed
                     INSIDE the gate (Phase 2, finding B2): should_promote takes
                     sigma and derives the margin itself, so the noise defense no
                     longer depends on the caller remembering to pass margin.
  2. cheap screen  — one dev pass + hard regression gate; kill obvious losers.
  3. paired confirm— re-evaluate survivors K times on a held-out set, PAIRED vs
                     the parent. Phase 2 (finding B1): uses the SAMPLE stdev and a
                     one-sided t critical value (df = n-1), not population stdev
                     with a fixed z=1.96 — the old form was anti-conservative for
                     the small K (3-5) this loop actually uses.
  4. lexicographic — correctness is a hard gate; efficiency only breaks ties.

Stdlib only. Swap in scipy/bootstrap for heavier use; the t-table below is the
one-sided 95% critical value t_{0.95, df}.
"""
from __future__ import annotations
import math
import statistics
from dataclasses import dataclass


def _finite(*vals: float) -> bool:
    """True only if every value is a real, finite number. The gate is the JUDGE:
    a non-finite score (NaN/inf from a broken measurement or an evolved solver)
    must FAIL CLOSED, never crash the loop or sneak a promotion through."""
    try:
        return all(isinstance(v, (int, float)) and math.isfinite(v) for v in vals)
    except TypeError:
        return False

# one-sided 95% t critical values, df -> t_{0.95, df}
_T95 = {1: 6.314, 2: 2.920, 3: 2.353, 4: 2.132, 5: 2.015, 6: 1.943, 7: 1.895,
        8: 1.860, 9: 1.833, 10: 1.812, 12: 1.782, 15: 1.753, 20: 1.725,
        25: 1.708, 30: 1.697}


def _t95(df: int) -> float:
    """One-sided 95% t critical value, interpolated/clamped from the table."""
    if df <= 0:
        return _T95[1]
    if df in _T95:
        return _T95[df]
    keys = sorted(_T95)
    if df > keys[-1]:
        return 1.645  # -> normal one-sided 95% as df -> inf
    lo = max(k for k in keys if k < df)
    hi = min(k for k in keys if k > df)
    frac = (df - lo) / (hi - lo)
    return _T95[lo] + frac * (_T95[hi] - _T95[lo])


@dataclass
class Score:
    correct: float       # fraction passing verification (the hard objective)
    efficiency: float    # tie-breaker only (tokens, time, cost) — higher = better


def noise_floor(baseline_scores: list[float]) -> float:
    """Sigma of the fixed baseline agent over repeated runs. Measure this FIRST."""
    return statistics.pstdev(baseline_scores) if len(baseline_scores) > 1 else 0.0


def passes_cheap_screen(candidate_dev: float, parent_dev: float,
                        regression_ok: bool) -> bool:
    return regression_ok and candidate_dev > parent_dev


def paired_confirm(candidate_runs: list[float], parent_runs: list[float],
                   margin: float = 0.0) -> bool:
    """
    Paired comparison on the SAME held-out instances. Promote only if the mean
    paired difference clears `margin` AND the one-sided 95% lower confidence bound
    is > 0, using sample stdev + t (df = n-1). `margin` plays the thresholdout
    role: raise it to spend holdout queries more slowly.
    """
    n = min(len(candidate_runs), len(parent_runs))
    if n == 0:
        return False
    diffs = [c - p for c, p in zip(candidate_runs[:n], parent_runs[:n])]
    if not _finite(margin, *candidate_runs[:n], *parent_runs[:n]):
        return False                        # fail closed on NaN/inf samples
    mean = statistics.fmean(diffs)
    if n == 1:
        return mean > margin
    sd = statistics.stdev(diffs)            # SAMPLE stdev (Bessel), not population
    if sd == 0.0:
        return mean > margin                # zero variance: decided by the margin
    se = sd / (n ** 0.5)
    ci_low = mean - _t95(n - 1) * se        # one-sided 95% lower bound
    return mean > margin and ci_low > 0.0


def lexicographic_better(cand: Score, parent: Score, eps: float = 1e-9) -> bool:
    """Correctness dominates; efficiency only matters at equal correctness."""
    if cand.correct > parent.correct + eps:
        return True
    if abs(cand.correct - parent.correct) <= eps:
        return cand.efficiency > parent.efficiency + eps
    return False


def should_promote(cand: Score, parent: Score, *,
                   candidate_dev: float, parent_dev: float, regression_ok: bool,
                   candidate_runs: list[float], parent_runs: list[float],
                   anchor_ok: bool, sigma: float = 0.0,
                   sigma_mult: float = 2.0) -> bool:
    """
    Full gate. The noise margin is derived INSIDE the gate from sigma (B2), so a
    caller can't silently disable noise protection by forgetting `margin`.
    anchor_ok = ground-truth anchor set did not regress (Goodhart guard).
    """
    if not anchor_ok:
        return False
    # JUDGE invariant: reject non-finite inputs outright (fail closed, no crash).
    if not _finite(cand.correct, cand.efficiency, parent.correct, parent.efficiency,
                   candidate_dev, parent_dev, sigma, sigma_mult,
                   *candidate_runs, *parent_runs):
        return False
    if not passes_cheap_screen(candidate_dev, parent_dev, regression_ok):
        return False
    margin = sigma_mult * sigma
    if not paired_confirm(candidate_runs, parent_runs, margin=margin):
        return False
    return lexicographic_better(cand, parent)
