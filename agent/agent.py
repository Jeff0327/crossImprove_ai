"""
agent/agent.py  —  PART OF THE GENOME (the loop may edit this file).

A SINGLE agent that can play either side of self-play: propose (with a verifier),
solve, critique, and rebut. Unifying the two roles into one class is what makes
role-swap actually expressible — generation N: A proposes / B solves; generation
N+1: B proposes / A solves. With separate Proposer/Solver classes that was
impossible (a Solver had no propose method), which was finding A3 in review.

Behavior is driven by two things:
  * `llm`  — an optional client exposing `.complete(prompt) -> str`. When None,
             a small deterministic fallback handles the toy arithmetic domain so
             the loop RUNS and is testable without a model.
  * `genome` — the evolvable state (see agent/genome.py). In a real run the LLM
             rewrites agent code; the genome knob lets the harness exercise the
             full mutate→measure→gate→commit cycle deterministically in tests.
"""
from __future__ import annotations
import re
from agent.types import Task
from agent.genome import Genome, Mutator


class Agent:
    def __init__(self, name: str, llm=None, genome: Genome | None = None):
        self.name = name
        self.llm = llm
        self.genome = genome or Genome()

    # ---- proposing side ----------------------------------------------------

    def propose(self, difficulty: float, seed: int) -> Task:
        """
        Emit a task + verifier. Prefer procedural generators (no fixed answer to
        memorize). The LLM path is for domains where templated generation is hard.
        """
        if self.llm is None:
            from benchmarks.procedural import example_math
            return example_math.generate(difficulty=difficulty, seed=seed)
        raise NotImplementedError(
            "LLM proposing not wired; pass a procedural generator or wire self.llm"
        )

    def critique(self, task: Task, solution: str, transcript: list[str]) -> str | None:
        """One concrete objection, or None when out of objections (verify, don't trust)."""
        if self.llm is None:
            # toy fallback: object once if the answer is obviously non-numeric
            already = any(t.startswith("OBJECTION") for t in transcript)
            if not already and not re.fullmatch(r"-?\d+", str(solution).strip()):
                return "solution is not an integer"
            return None
        raise NotImplementedError("LLM critique not wired")

    # ---- solving side ------------------------------------------------------

    def solve(self, task: Task) -> str:
        """Produce a candidate solution in the format the verifier expects."""
        if self.llm is None:
            # toy fallback for the arithmetic domain: read the operands and add.
            # genome.strategy DRIVES behavior so mutation can change fitness:
            #   "naive"   -> off-by-one bug (wrong) ; "correct" -> right answer.
            nums = re.findall(r"-?\d+", task.prompt)
            if len(nums) >= 2 and "+" in task.prompt:
                total = int(nums[0]) + int(nums[1])
                if self.genome.strategy == "naive":
                    total += 1            # the bug a mutation can fix
                return str(total)
            return ""  # unknown domain -> empty (will fail verification, honestly)
        raise NotImplementedError("LLM solving not wired")

    def rebut(self, task: Task, solution: str, objection: str,
              transcript: list[str]) -> str:
        """Respond to an objection; may revise. Re-verified afterwards."""
        if self.llm is None:
            return self.solve(task)  # toy: just re-derive
        raise NotImplementedError("LLM rebuttal not wired")

    # ---- self-improvement seam --------------------------------------------

    def self_improve(self, mutator: Mutator) -> "Agent":
        """
        Produce a child agent with a mutated genome. This is the single seam where
        real evolution happens: a real Mutator asks the LLM to rewrite agent code;
        the ToyMutator perturbs the genome knob so the harness is exercisable.
        """
        child_genome = mutator.mutate(self.genome)
        return Agent(name=self.name, llm=self.llm, genome=child_genome)
