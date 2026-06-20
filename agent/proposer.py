"""
agent/proposer.py  —  PART OF THE GENOME (the loop may edit this file).

The proposer's job is NOT to decide whether an answer is correct. Its job is to
emit a problem together with a *verification script* that decides correctness by
execution. This is the single most important design choice in the whole system:
a model that judges can only offer an opinion; a model that writes a checker
produces something that fails objectively when the answer is wrong.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Task:
    """A self-contained, auto-checkable problem."""
    prompt: str            # natural-language problem statement given to the solver
    verifier_src: str      # python source defining `def check(solution: str) -> bool`
    meta: dict             # difficulty params, domain tag, provenance, seed, etc.


class Proposer:
    def __init__(self, llm, domain: str):
        self.llm = llm          # any object exposing .complete(prompt) -> str
        self.domain = domain

    def propose(self, difficulty: float, seed: int) -> Task:
        """
        Emit a fresh task. For verifiable domains, prefer PROCEDURAL generation
        (see benchmarks/procedural) so there is no fixed answer to memorize.
        The LLM path below is for domains where templated generation is hard.
        """
        # TODO(llm): prompt the model to produce BOTH a problem and a checker.
        # The checker must be pure, deterministic, and import-light. Example
        # contract the prompt should enforce:
        #
        #   Return JSON: {"prompt": "...", "verifier": "def check(solution):..."}
        #   - check() returns True only for correct solutions
        #   - check() must not read the network, files, or env
        #   - prefer numeric/exact comparison over fuzzy matching
        #
        raise NotImplementedError("wire self.llm here, or use a procedural generator")

    def critique(self, task: Task, solution: str, transcript: list[str]) -> str | None:
        """
        Debate move: raise ONE concrete objection to the solver's answer, or
        return None if you have no further objection. Remember: running out of
        objections is not proof of correctness — the execution verifier, not this
        method, casts the verdict. Keep objections specific and checkable.
        """
        # TODO(llm): ask the model for the single strongest objection, grounded
        # in the problem. Return None to signal "no further objection".
        raise NotImplementedError
