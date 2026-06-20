"""
orchestrator/loop.py

The main self-play loop. One pass = one generation:

    sample parent (with diversity) -> propose task + verifier -> solve -> debate
    -> execution verdict -> gate (double eval + anchors) -> promote? -> git push
    -> archive -> ROLE SWAP -> repeat.

Everything load-bearing lives in runner/ (the judge) which this file calls but
does not modify. The agent/ objects ARE the genome; promoting means committing
their changed source to git.

This is a runnable skeleton: wire a real LLM into agent/* and a real git remote,
then flesh out the marked steps. It is deliberately small so the control flow is
readable end to end.
"""
from __future__ import annotations
import subprocess

from agent.proposer import Proposer
from agent.solver import Solver
from agent.debate import run_debate
from runner import verifier, gate
from benchmarks.procedural import example_math


# ---- git as genome substrate ------------------------------------------------

def git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout

def commit_and_push(message: str) -> None:
    git("add", "agent")          # only the genome is committed by the loop
    git("commit", "-m", message)
    git("push")                  # configure the remote/branch out of band

def sample_parent(archive: list) -> object:
    """
    Diversity sampling, NOT greedy-best. Every archived agent keeps a non-zero
    selection probability (better ones higher). Greedy selection collapses into
    local optima; the archive is what makes exploration open-ended.
    """
    # TODO: weight by score while keeping all probabilities > 0 (e.g. softmax).
    return archive[-1] if archive else None


# ---- one generation ---------------------------------------------------------

def generation(proposer: Proposer, solver: Solver, *, parent_score,
               anchor_ok: bool, sigma: float) -> bool:
    # ① fresh, procedurally generated task (no fixed answer to memorize)
    task = example_math.generate(difficulty=1.0, seed=None)

    # ② solve, then ③ debate to refine the candidate
    solution = solver.solve(task)
    solution, _transcript = run_debate(proposer, solver, task, solution)

    # ④ the verdict is execution, never a model's say-so
    if not verifier.judge(task, solution).ok:
        return False

    # ⑤ promotion gate: noise floor + paired confirm + anchors + lexicographic
    cand = gate.Score(correct=1.0, efficiency=0.0)   # fill from real measurements
    promote = gate.should_promote(
        cand, parent_score,
        candidate_dev=1.0, parent_dev=0.0, regression_ok=True,
        candidate_runs=[1.0], parent_runs=[0.0],
        anchor_ok=anchor_ok, margin=2 * sigma,       # 2*sigma = beat the noise
    )
    if promote:
        commit_and_push("promote: candidate cleared gate")
    return promote


def main(generations: int = 100) -> None:
    # measure the noise floor ONCE before optimizing anything
    sigma = 0.0  # TODO: run the fixed baseline ~30x on the anchor set, take pstdev

    llm = None   # TODO(llm): a local model client exposing .complete(prompt)
    a, b = Proposer(llm, domain="arithmetic"), Solver(llm)
    parent_score = gate.Score(correct=0.0, efficiency=0.0)

    for g in range(generations):
        # role swap every generation: proposer and solver trade places
        if g % 2 == 0:
            proposer, solver = a, b
        else:
            proposer, solver = Proposer(llm, domain="arithmetic"), Solver(llm)

        anchor_ok = True  # TODO: evaluate the anchor set; False if it regressed
        promoted = generation(proposer, solver,
                              parent_score=parent_score,
                              anchor_ok=anchor_ok, sigma=sigma)
        print(f"gen {g}: {'PROMOTED' if promoted else 'kept-in-archive'}")


if __name__ == "__main__":
    main()
