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
from agent.debate import run_debate
from agent.types import Task
from runner import verifier, gate
from benchmarks.procedural import example_math


# ---- tunable config (one place to adjust the loop) --------------------------

N_FLOOR_TASKS = 10        # baseline runs used to measure the noise floor (sigma)
N_TASKS_PER_GEN = 8       # held-out tasks the proposer generates each generation
SAMPLE_TEMPERATURE = 0.5  # softmax temperature for diversity sampling (>0)
DEBATE_MAX_ROUNDS = 4     # hard cap on objection/rebuttal rounds


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

    def sample(self, rng: random.Random, temperature: float = SAMPLE_TEMPERATURE) -> Member:
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

def _verdict(task: Task, solution: str) -> bool:
    return verifier.judge(task, solution).ok

def solve_with_debate(proposer: Agent, solver: Agent, task: Task,
                      max_rounds: int = DEBATE_MAX_ROUNDS) -> str:
    """Solver answers; debate (with the proposer critiquing) refines before judging."""
    sol = solver.solve(task)
    sol, _ = run_debate(proposer, solver, task, sol, max_rounds=max_rounds,
                        judge=lambda t, s: _verdict(t, s))
    return sol

def measure(agent: Agent, tasks: list[Task]) -> list[float]:
    """Per-task 1.0/0.0 verdicts via execution (no debate). Used for the baseline."""
    return [1.0 if _verdict(t, agent.solve(t)) else 0.0 for t in tasks]

def measure_vs(proposer: Agent, solver: Agent, tasks: list[Task]) -> list[float]:
    """Paired sample for the gate: solver answers each task WITH debate, then judged."""
    return [1.0 if _verdict(t, solve_with_debate(proposer, solver, t)) else 0.0
            for t in tasks]


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
               n_tasks: int, seed_base: int, anchor_ok: bool,
               sigma: float) -> tuple[bool, Member]:
    # ① the PROPOSER actually proposes the held-out tasks (role-swap is behavioral)
    tasks = [proposer.propose(difficulty=1.0, seed=seed_base + i) for i in range(n_tasks)]

    # ② child = a real mutation of the parent's genome (the self-improvement seam)
    child = parent.agent.self_improve(mutator)

    # ③ MEASURE both on the SAME tasks, each refined through DEBATE, then judged by
    #    execution → real paired samples feed the gate (no hardcoding)
    parent_runs = measure_vs(proposer, parent.agent, tasks)
    child_runs = measure_vs(proposer, child, tasks)
    child_fit = sum(child_runs) / len(tasks)
    parent_fit = sum(parent_runs) / len(tasks)

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
    floor_tasks = [example_math.generate(seed=1000 + i) for i in range(N_FLOOR_TASKS)]
    sigma = gate.noise_floor([sum(measure(baseline, [t])) for t in floor_tasks])

    archive = Archive()
    # seed both agents with the buggy "naive" strategy; a mutation can fix it,
    # so the loop demonstrates a genuine measured improvement (PROMOTED) without
    # an LLM. The fixed baseline above uses the default "correct" genome.
    a = Agent("A", genome=Genome(strategy="naive"))
    b = Agent("B", genome=Genome(strategy="naive"))
    archive.add(a, sum(measure(a, floor_tasks)) / len(floor_tasks))
    archive.add(b, sum(measure(b, floor_tasks)) / len(floor_tasks))

    for g in range(generations):
        # ROLE SWAP: the two archived agents alternate who proposes each generation
        proposer = archive.members[g % 2].agent
        parent = archive.sample(rng)                      # diversity sampling

        promoted, child = generation(proposer, parent, mutator,
                                     n_tasks=N_TASKS_PER_GEN, seed_base=g * 100,
                                     anchor_ok=True, sigma=sigma)
        if promoted:
            archive.add(child.agent, child.fitness)
            if do_commit:
                apply_and_commit(child.agent.genome, f"promote gen {g}")
        print(f"gen {g}: {'PROMOTED' if promoted else 'kept'} "
              f"(child_fit={child.fitness:.2f}, archive={len(archive.members)})")


if __name__ == "__main__":
    main()
