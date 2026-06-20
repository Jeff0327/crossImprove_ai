"""
runner/gate.py  —  THE JUDGE. The loop must NOT edit this file.

The promotion gate. Stops the loop from pushing NOISE as if it were improvement.
Four stacked checks, cheapest first:

  1. noise floor   — measure baseline score variance (sigma) ONCE up front; if a
                     generation's gain is below ~2*sigma it is noise, not signal.
  2. cheap screen  — one dev pass + hard regression gate; kill obvious losers.
  3. paired confirm— re-evaluate survivors K times on a held-out set, PAIRED vs
                     the parent, and promote only if the CI lower bound > 0
                     (with a thresholdout-style margin to slow holdout burn-in).
  4. lexicographic — correctness is a hard gate; efficiency only breaks ties.
                     Never let the optimizer trade accuracy for speed.

This module intentionally uses only the stdlib so the skeleton runs anywhere;
swap in numpy/scipy for real bootstrap CIs and McNemar tests.
"""
from __future__ import annotations
import statistics
from dataclasses import dataclass


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
    paired difference clears `margin` and the (rough) CI lower bound is > 0.
    `margin` plays the thresholdout role: raise it to spend holdout queries more
    slowly and resist selection-driven overfitting to the holdout.
    """
    n = min(len(candidate_runs), len(parent_runs))
    if n == 0:
        return False
    diffs = [c - p for c, p in zip(candidate_runs[:n], parent_runs[:n])]
    mean = statistics.fmean(diffs)
    if n == 1:
        return mean > margin
    sd = statistics.pstdev(diffs)
    ci_low = mean - 1.96 * (sd / (n ** 0.5))   # replace with bootstrap for real use
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
                   anchor_ok: bool, margin: float = 0.0) -> bool:
    """Full gate. anchor_ok = ground-truth anchor set did not regress."""
    if not anchor_ok:
        return False  # Goodhart / late-stage collapse signal — refuse promotion
    if not passes_cheap_screen(candidate_dev, parent_dev, regression_ok):
        return False
    if not paired_confirm(candidate_runs, parent_runs, margin=margin):
        return False
    return lexicographic_better(cand, parent)
