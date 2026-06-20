"""
benchmarks/procedural/example_math.py

A worked example of the anti-memorization principle: every call emits a FRESH
instance plus a checker. There is no fixed answer to hardcode, so an evolved
patch that memorizes answers gains nothing — memorization stops being a winning
strategy. This file is a template; the real value is in the pattern, not the
specific math. Swap in algorithm/optimization/coding generators the same way.
"""
from __future__ import annotations
import random
from agent.proposer import Task


def generate(difficulty: float = 1.0, seed: int | None = None) -> Task:
    rng = random.Random(seed)
    # difficulty scales the coefficient range -> keeps the task near the agent's
    # ability so the score has headroom (discrimination, IRT-style).
    hi = int(5 + 20 * difficulty)
    a = rng.randint(2, hi)
    b = rng.randint(2, hi)
    answer = a + b  # ground truth computed here, never shipped to the solver

    prompt = f"Compute {a} + {b}. Respond with the integer only."

    # The verifier embeds the answer, runs in an isolated process, and is the sole
    # arbiter. It is generated fresh with the instance, so it cannot be memorized
    # across generations.
    verifier_src = (
        "def check(solution):\n"
        "    try:\n"
        "        return int(str(solution).strip()) == %d\n"
        "    except Exception:\n"
        "        return False\n" % answer
    )

    return Task(
        prompt=prompt,
        verifier_src=verifier_src,
        meta={"domain": "arithmetic", "difficulty": difficulty,
              "seed": seed, "provenance": "procedural"},
    )
