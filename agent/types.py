"""agent/types.py — shared data types (no logic, no LLM)."""
from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class Task:
    """A self-contained, auto-checkable problem."""
    prompt: str            # problem statement shown to the solver
    verifier_src: str      # python source defining `def check(solution: str) -> bool`
    meta: dict = field(default_factory=dict)
