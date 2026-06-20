"""
runner/verifier.py  —  THE JUDGE. The loop must NOT edit this file.

Casts the verdict by execution, and — critically — audits ITSELF. An imperfect
verifier is a gap the optimizer will find and exploit. So we periodically feed
the verifier known-bad solutions and measure how many it wrongly accepts
(false-negative rate on catching errors). A rising gap is an early warning that
upcoming "improvements" may be exploits.
"""
from __future__ import annotations
from dataclasses import dataclass
from agent.types import Task
from runner import sandbox


@dataclass
class Verdict:
    ok: bool
    detail: dict


def judge(task: Task, solution: str) -> Verdict:
    """The verdict is execution, not opinion."""
    res = sandbox.verify(task.verifier_src, solution)
    return Verdict(ok=res.get("ok", False), detail=res)


def audit_verifier(task: Task, known_bad: list[str]) -> float:
    """
    Adversarial check on the verifier itself. `known_bad` are solutions known to
    be WRONG. Returns the fraction the verifier wrongly accepted (should be ~0).
    Spikes here = the generator-verifier gap is widening; tighten the checker
    before trusting further score gains.
    """
    if not known_bad:
        return 0.0
    wrongly_accepted = sum(1 for s in known_bad if judge(task, s).ok)
    return wrongly_accepted / len(known_bad)
