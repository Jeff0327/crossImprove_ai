"""
agent/genome.py — the evolvable state, and the Mutator seam.

In a real run the LLM rewrites agent code and the diff IS the mutation. To make
the loop runnable and testable WITHOUT a model, we expose a tiny tunable genome
and a ToyMutator that makes a real, committable change. The LLMMutator is the
seam you fill to get genuine code-evolution; it fails closed until wired.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict
import json
import pathlib

GENOME_PATH = pathlib.Path(__file__).with_name("genome.json")


@dataclass
class Genome:
    # toy, documented knobs that ACTUALLY drive Agent.solve so the loop can
    # demonstrate a real fitness gain without an LLM. strategy:
    #   "naive"   -> buggy solver (off-by-one) -> low fitness
    #   "correct" -> fixed solver              -> high fitness
    revision: int = 0
    strategy: str = "correct"

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def save(self, path: pathlib.Path = GENOME_PATH) -> None:
        path.write_text(self.to_json() + "\n")

    @classmethod
    def load(cls, path: pathlib.Path = GENOME_PATH) -> "Genome":
        if path.exists():
            return cls(**json.loads(path.read_text()))
        return cls()


class Mutator:
    """Interface. mutate() returns a NEW genome (never mutates in place)."""
    def mutate(self, genome: Genome) -> Genome:
        raise NotImplementedError


class ToyMutator(Mutator):
    """
    Deterministic, real change. If the parent is "naive" it fixes the bug
    ("correct") — a genuine, measurable improvement the gate will promote. Once
    correct, it just bumps revision (a real diff, but no fitness change), so the
    loop converges and then stops promoting noise — exactly the intended shape.
    """
    def mutate(self, genome: Genome) -> Genome:
        if genome.strategy == "naive":
            return Genome(revision=genome.revision + 1, strategy="correct")
        return Genome(revision=genome.revision + 1, strategy=genome.strategy)


class LLMMutator(Mutator):
    """The real seam: ask the LLM to rewrite agent code. Fails closed until wired."""
    def __init__(self, llm):
        self.llm = llm

    def mutate(self, genome: Genome) -> Genome:
        # TODO(llm): prompt the model for a patch to agent/*.py, apply it, and
        # return a genome describing the new revision. Must fail closed if no
        # valid, verifier-passing patch is produced.
        raise NotImplementedError("LLM-driven genome mutation not wired")
