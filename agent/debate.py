"""
agent/debate.py  —  PART OF THE GENOME (the loop may edit this file).

Bounded debate. Debate is a truth AMPLIFIER, not a truth detector: it refines the
candidate so the execution verifier gets a cleaner thing to judge. It never issues
the verdict itself. Two guards keep it from degenerating:

  * a hard round cap (debate that never ends is debate that learned nothing), and
  * the verdict is deferred to runner/verifier.py regardless of how debate "felt".
"""
from __future__ import annotations
from agent.proposer import Proposer, Task
from agent.solver import Solver


def run_debate(proposer: Proposer, solver: Solver, task: Task,
               solution: str, max_rounds: int = 4) -> tuple[str, list[str]]:
    """
    Returns the refined solution and the full transcript. Stops early when the
    proposer raises no further objection — but see the caveat above: "no further
    objection" is a signal to verify, not a certificate of correctness.
    """
    transcript: list[str] = [f"SOLUTION: {solution}"]
    for _ in range(max_rounds):
        objection = proposer.critique(task, solution, transcript)
        if objection is None:
            break
        transcript.append(f"OBJECTION: {objection}")
        solution = solver.rebut(task, solution, objection, transcript)
        transcript.append(f"REBUTTAL/SOLUTION: {solution}")
    return solution, transcript
