"""
agent/debate.py  —  PART OF THE GENOME (the loop may edit this file).

Bounded debate between two Agents. Truth AMPLIFIER, not detector: it refines the
candidate so the execution verifier gets a cleaner thing to judge; it never casts
the verdict. Guards: a hard round cap, and — new in Phase 1 — a MONOTONIC guard so
a rebuttal can never hand back a candidate worse than the best one seen, which was
finding F in review (debate could regress a correct answer).
"""
from __future__ import annotations
from agent.types import Task
from agent.agent import Agent


def run_debate(proposer: Agent, solver: Agent, task: Task, solution: str,
               max_rounds: int = 4, judge=None) -> tuple[str, list[str]]:
    """
    Returns (refined_solution, transcript). If a `judge(task, sol) -> bool` is
    given, the monotonic guard keeps the last solution that passed; a regressing
    rebuttal is discarded. Without a judge it behaves as before (last solution).
    """
    transcript: list[str] = [f"SOLUTION: {solution}"]
    best = solution
    best_ok = judge(task, solution) if judge else None
    for _ in range(max_rounds):
        objection = proposer.critique(task, solution, transcript)
        if objection is None:
            break
        transcript.append(f"OBJECTION: {objection}")
        solution = solver.rebut(task, solution, objection, transcript)
        transcript.append(f"REBUTTAL/SOLUTION: {solution}")
        if judge is not None:
            now_ok = judge(task, solution)
            # keep the new one only if it didn't regress a passing candidate
            if now_ok or not best_ok:
                best, best_ok = solution, now_ok
            else:
                solution = best  # discard regressing rebuttal
        else:
            best = solution
    return best, transcript
