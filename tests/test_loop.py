"""
tests/test_loop.py

Behavioral tests for the skeleton. These check that the DESIGN INTENT holds in
code, not just that modules import:

  * procedural generation stays fresh (anti-memorization),
  * the verdict comes from execution and survives hostile verifiers,
  * the verifier can audit its own generator-verifier gap,
  * the gate refuses noise and refuses promotion when anchors regress (Goodhart),
  * debate is bounded and defers the verdict,
  * one full generation runs end-to-end with a mock LLM,
  * unwired stubs fail closed.

No LLM or network needed — runs on the stdlib via pytest.
"""
import re
import pytest

from agent.proposer import Task, Proposer
from agent.solver import Solver
from agent.debate import run_debate
from runner import sandbox, verifier, gate
from benchmarks.procedural import example_math

S = gate.Score


def _truth(task: Task) -> str:
    a, b = map(int, re.findall(r"\d+", task.prompt)[:2])
    return str(a + b)


# --- [1] procedural generator ------------------------------------------------

def test_same_seed_reproducible():
    assert example_math.generate(seed=1).prompt == example_math.generate(seed=1).prompt

def test_different_seed_differs():
    assert example_math.generate(seed=1).prompt != example_math.generate(seed=2).prompt

def test_provenance_tagged():
    assert example_math.generate(seed=1).meta["provenance"] == "procedural"

def test_difficulty_scales_range():
    hard = max(int(example_math.generate(difficulty=5.0, seed=i).prompt.split()[1]) for i in range(50))
    easy = max(int(example_math.generate(difficulty=0.1, seed=i).prompt.split()[1]) for i in range(50))
    assert hard > easy


# --- [2] verdict by execution ------------------------------------------------

@pytest.fixture
def task():
    return example_math.generate(seed=1)

def test_correct_accepted(task):
    assert verifier.judge(task, _truth(task)).ok is True

@pytest.mark.parametrize("bad", ["-999", "twenty six", ""])
def test_wrong_rejected(task, bad):
    assert verifier.judge(task, bad).ok is False


# --- [3] sandbox robustness against hostile verifiers ------------------------

def test_infinite_loop_times_out():
    res = sandbox.verify("def check(s):\n    while True: pass", "x", timeout=2.0)
    assert res["ok"] is False and "timeout" in res.get("error", "")

def test_raising_verifier_caught():
    assert sandbox.verify("def check(s):\n    raise ValueError('x')", "x")["ok"] is False

def test_sysexit_in_verifier_handled():
    assert sandbox.verify("def check(s):\n    import sys; sys.exit(3)", "x")["ok"] is False


# --- [4] verifier self-audit (generator-verifier gap) ------------------------

def test_loose_verifier_flagged():
    loose = Task("p", "def check(s): return True", {})
    assert verifier.audit_verifier(loose, known_bad=["a", "b", "c"]) == 1.0

def test_strict_verifier_clean():
    strict = Task("p", "def check(s): return s=='ok'", {})
    assert verifier.audit_verifier(strict, known_bad=["x", "y"]) == 0.0


# --- [5] gate: double eval ---------------------------------------------------

def test_lexicographic_correctness_dominates():
    assert gate.lexicographic_better(S(0.9, 0.1), S(0.8, 0.9))

def test_lexicographic_efficiency_breaks_ties_only():
    assert gate.lexicographic_better(S(0.8, 0.9), S(0.8, 0.5))

def test_lexicographic_never_trades_accuracy_for_speed():
    assert not gate.lexicographic_better(S(0.7, 0.99), S(0.8, 0.0))

def test_noise_floor():
    assert gate.noise_floor([0.5, 0.5, 0.5]) == 0.0
    assert gate.noise_floor([0.0, 1.0]) > 0

def test_cheap_screen_requires_regression_ok():
    assert gate.passes_cheap_screen(0.9, 0.8, True)
    assert not gate.passes_cheap_screen(0.9, 0.8, False)

def test_paired_confirm_rejects_noise():
    assert not gate.paired_confirm([0.51, 0.49, 0.50], [0.50, 0.50, 0.50])

def test_paired_confirm_accepts_clear_win():
    assert gate.paired_confirm([0.9] * 4, [0.5] * 4)

def test_anchor_regression_blocks_promotion():
    blocked = gate.should_promote(
        S(1.0, 0.0), S(0.0, 0.0), candidate_dev=1.0, parent_dev=0.0,
        regression_ok=True, candidate_runs=[1] * 4, parent_runs=[0] * 4,
        anchor_ok=False, margin=0.0)
    assert blocked is False

def test_clean_improvement_promotes():
    ok = gate.should_promote(
        S(1.0, 0.0), S(0.0, 0.0), candidate_dev=1.0, parent_dev=0.0,
        regression_ok=True, candidate_runs=[1] * 4, parent_runs=[0] * 4,
        anchor_ok=True, margin=0.0)
    assert ok is True


# --- [6] debate: bounded, defers verdict -------------------------------------

class _StoppingProposer:
    def __init__(self): self.calls = 0
    def critique(self, task, sol, tr):
        self.calls += 1
        return "off by one?" if self.calls < 3 else None

class _ConvergingSolver:
    def rebut(self, task, sol, obj, tr): return _truth(task)

def test_debate_converges_and_stops_early(task):
    p = _StoppingProposer()
    final, transcript = run_debate(p, _ConvergingSolver(), task, "bad-guess", max_rounds=4)
    assert final == _truth(task)
    assert p.calls == 3
    assert any("OBJECTION" in x for x in transcript)

def test_debate_respects_hard_cap(task):
    class _Relentless:
        def critique(self, *a): return "still wrong"
    _, tr = run_debate(_Relentless(), _ConvergingSolver(), task, "x", max_rounds=3)
    assert sum(1 for x in tr if x.startswith("OBJECTION")) == 3


# --- [7] full generation with a mock LLM -------------------------------------

def test_full_generation_end_to_end():
    class _P:
        def critique(self, task, sol, tr): return None
    class _Sv:
        def solve(self, task): return _truth(task)
        def rebut(self, task, sol, obj, tr): return sol
    t = example_math.generate(seed=7)
    sol = _Sv().solve(t)
    sol, _ = run_debate(_P(), _Sv(), t, sol)
    assert verifier.judge(t, sol).ok is True
    promoted = gate.should_promote(
        S(1.0, 0.0), S(0.0, 0.0), candidate_dev=1.0, parent_dev=0.0,
        regression_ok=True, candidate_runs=[1, 1, 1], parent_runs=[0, 0, 0],
        anchor_ok=True, margin=0.0)
    assert promoted is True


# --- [8] stubs fail closed ---------------------------------------------------

def test_proposer_stub_fails_closed():
    with pytest.raises(NotImplementedError):
        Proposer(llm=None, domain="x").propose(1.0, 0)

def test_solver_stub_fails_closed():
    with pytest.raises(NotImplementedError):
        Solver(llm=None).solve(example_math.generate(seed=1))
