"""
agent/solver.py  —  PART OF THE GENOME (the loop may edit this file).

The solver produces a candidate solution and defends it across debate rounds.
When a candidate survives the debate AND passes execution verification AND clears
the gate, the orchestrator upgrades this agent (weights or scaffolding) and the
change is committed to the genome.
"""
from __future__ import annotations
from agent.proposer import Task


class Solver:
    def __init__(self, llm):
        self.llm = llm

    def solve(self, task: Task) -> str:
        """Produce a candidate solution to task.prompt."""
        # TODO(llm): prompt the model with task.prompt. Return the solution in the
        # exact format the verifier's check() expects (the proposer defines it).
        raise NotImplementedError

    def rebut(self, task: Task, solution: str, objection: str,
              transcript: list[str]) -> str:
        """
        Debate move: respond to a single objection. May revise the solution.
        Return the (possibly updated) solution. The loop re-verifies after the
        debate, so honesty beats rhetoric — a rhetorical win that fails check()
        is discarded.
        """
        # TODO(llm): address the objection; return the current best solution.
        raise NotImplementedError
