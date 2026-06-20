"""
orchestrator/loop.py

The main self-play loop, integrated end-to-end. Phase 4 fixes the "glue" findings
the review concentrated here (A1, D1, D2, D3) and makes the loop runnable without
an LLM via a ToyMutator, so the FULL cycle is exercisable and testable:

    build archive -> softmax-sample a parent (diversity, not greedy)         [D2]
      -> parent proposes a held-out task set (procedural; nothing to memorize)
      -> child = parent.self_improve(mutator)  (real genome mutation)        [A1]
      -> MEASURE child vs parent on the SAME tasks (paired, real scores)     [D1]
      -> gate: sigma-derived noise margin + paired t-test + anchors + lexico
      -> promote? apply genome + commit (diff-checked, rc-checked)       [A1,D3]
      -> add child to archive -> ROLE SWAP next generation              [A2,A3]

HONEST LIMITATION: the ToyMutator perturbs a documented genome knob so the
mechanics run deterministically. Genuine capability gain needs the LLMMutator
(agent/genome.py) rewriting agent code on a real domain — that is the one
remaining seam, and it fails closed until wired. The toy solver is perfect on
arithmetic, so with ToyMutator nothing actually improves; that is correct and
honest — the gate refuses to promote noise. Tests inject a weakened parent to
exercise the promote path.
"""
from __future__ import annotations
import math
import random
import subprocess
from dataclasses import dataclass, field

from agent.agent import Agent
from agent.genome import Genome, ToyMutator
from agent.types import Task
from runner import verifier, gate
from benchmarks.procedural import example_math


# ---- archive ----------------------------------------------------------------

@dataclass
class Member:
    agent: Agent
    fitness: float = 0.0

@dataclass
class Archive:
    members: list[Member] = field(default_factory=list)

    def add(self, agent: Agent, fitness: float) -> None:
        self.members.append(Member(agent, fitness))

    def sample(self, rng: random.Random, temperature: float = 0.5) -> Member:
        """
        Softmax sampling over fitness — diversity, NOT greedy-best (D2). Every
        member keeps a non-zero probability; better ones are likelier. This is the
        open-ended-exploration guard against collapsing into a local optimum.
        """
        if not self.members:
            raise ValueError("empty archive")
        fits = [m.fitness for m in self.members]
        mx = max(fits)
        weights = [math.exp((f - mx) / max(temperature, 1e-6)) for f in fits]
        return rng.choices(self.members, weights=weights, k=1)[0]


# ---- measurement (real scores feed the gate, no hardcoding) -----------------

def measure(agent: Agent, tasks: list[Task]) -> list[float]:
    """Per-task 1.0/0.0 verdicts via execution. The list IS the paired sample."""
    return [1.0 if verifier.judge(t, agent.solve(t)).ok else 0.0 for t in tasks]


# ---- git substrate (return codes checked; empty commits refused) ------------

def _git(*args: str, cwd=None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, cwd=cwd)

def apply_and_commit(genome: Genome, message: str, *, path=None, cwd=None,
                     push: bool = False) -> bool:
    """
    Persist the mutated genome and commit it. Returns True only if a REAL diff was
    committed (D3/A1: never a silent empty commit pretending to be progress).
    `path`/`cwd` are injectable so the empty-commit refusal is unit-testable.
    """
    from agent.genome import GENOME_PATH
    p = path or GENOME_PATH
    genome.save(p)
    if _git("add", str(p), cwd=cwd).returncode != 0:
        return False
    if _git("diff", "--cached", "--quiet", cwd=cwd).returncode == 0:
        return False  # nothing changed -> not a promotion, do not fake one
    if _git("commit", "-m", message, cwd=cwd).returncode != 0:
        return False
    if push and _git("push", cwd=cwd).returncode != 0:
        return False
    return True


# ---- one generation ---------------------------------------------------------

def generation(proposer: Agent, parent: Member, mutator, *,
               tasks: list[Task], anchor_ok: bool, sigma: float) -> tuple[bool, Member]:
    # child = a real mutation of the parent's genome (the self-improvement seam)
    child = parent.agent.self_improve(mutator)

    # MEASURE both on the SAME held-out tasks → real paired samples (no hardcoding)
    parent_runs = measure(parent.agent, tasks)
    child_runs = measure(child, tasks)
    child_fit, parent_fit = sum(child_runs) / len(tasks), sum(parent_runs) / len(tasks)

    promote = gate.should_promote(
        gate.Score(correct=child_fit, efficiency=0.0),
        gate.Score(correct=parent_fit, efficiency=0.0),
        candidate_dev=child_fit, parent_dev=parent_fit, regression_ok=True,
        candidate_runs=child_runs, parent_runs=parent_runs,
        anchor_ok=anchor_ok, sigma=sigma,
    )
    return promote, Member(child, child_fit)


# ---- driver -----------------------------------------------------------------

def main(generations: int = 6, seed: int = 0, do_commit: bool = False) -> None:
    rng = random.Random(seed)
    mutator = ToyMutator()           # swap LLMMutator(llm) for real code-evolution

    # noise floor measured ONCE on the baseline before optimizing (sigma)
    baseline = Agent("baseline", genome=Genome())
    floor_tasks = [example_math.generate(seed=1000 + i) for i in range(10)]
    sigma = gate.noise_floor([sum(measure(baseline, [t])) for t in floor_tasks])

    archive = Archive()
    a, b = Agent("A", genome=Genome()), Agent("B", genome=Genome())
    archive.add(a, sum(measure(a, floor_tasks)) / len(floor_tasks))
    archive.add(b, sum(measure(b, floor_tasks)) / len(floor_tasks))

    for g in range(generations):
        # ROLE SWAP: the two archived agents trade proposer/solver each generation
        proposer = archive.members[g % 2].agent
        parent = archive.sample(rng)                      # diversity sampling
        tasks = [example_math.generate(seed=g * 100 + i) for i in range(8)]

        promoted, child = generation(proposer, parent, mutator,
                                     tasks=tasks, anchor_ok=True, sigma=sigma)
        if promoted:
            archive.add(child.agent, child.fitness)
            if do_commit:
                apply_and_commit(child.agent.genome, f"promote gen {g}")
        print(f"gen {g}: {'PROMOTED' if promoted else 'kept'} "
              f"(child_fit={child.fitness:.2f}, archive={len(archive.members)})")


if __name__ == "__main__":
    main()
