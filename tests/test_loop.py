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

from agent.types import Task
from agent.agent import Agent
from agent.genome import Genome, ToyMutator, LLMMutator
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
        anchor_ok=False, sigma=0.0)
    assert blocked is False

def test_clean_improvement_promotes():
    ok = gate.should_promote(
        S(1.0, 0.0), S(0.0, 0.0), candidate_dev=1.0, parent_dev=0.0,
        regression_ok=True, candidate_runs=[1] * 4, parent_runs=[0] * 4,
        anchor_ok=True, sigma=0.0)
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
        anchor_ok=True, sigma=0.0)
    assert promoted is True


# --- [8] stubs fail closed ---------------------------------------------------

def test_llm_paths_fail_closed():
    # with an llm set, the unwired LLM paths must fail closed (not silently pass)
    a = Agent("A", llm=object())
    with pytest.raises(NotImplementedError):
        a.propose(1.0, 0)
    with pytest.raises(NotImplementedError):
        a.solve(example_math.generate(seed=1))

def test_llm_mutator_fails_closed():
    with pytest.raises(NotImplementedError):
        LLMMutator(llm=object()).mutate(Genome())


# --- [9] Phase 1: unified Agent enables real role-swap -----------------------

def test_one_agent_plays_both_roles():
    from agent.agent import Agent
    a = Agent("A")
    t = a.propose(difficulty=1.0, seed=3)          # same object proposes...
    sol = a.solve(t)                               # ...and solves
    assert verifier.judge(t, sol).ok is True

def test_role_swap_uses_same_objects_both_sides():
    from agent.agent import Agent
    a, b = Agent("A"), Agent("B")
    t = a.propose(difficulty=1.0, seed=4)
    sol = b.solve(t)
    assert verifier.judge(t, sol).ok is True
    # swapped generation: b proposes, a solves — both are Agents, so this is valid
    t2 = b.propose(difficulty=1.0, seed=5)
    sol2 = a.solve(t2)
    assert verifier.judge(t2, sol2).ok is True

def test_debate_monotonic_guard_discards_regression():
    from agent.agent import Agent
    t = example_math.generate(seed=6)
    correct = verifier.judge
    good = None
    import re as _re
    good = str(sum(int(x) for x in _re.findall(r"\d+", t.prompt)[:2]))
    class _RegressingSolver:
        def __init__(self): self.n = 0
        def rebut(self, task, sol, obj, tr):
            self.n += 1
            return "-999"  # always regress to wrong
    class _NagProposer:
        def __init__(self): self.n = 0
        def critique(self, task, sol, tr):
            self.n += 1
            return "are you sure?" if self.n < 2 else None
    from agent.debate import run_debate
    final, _ = run_debate(_NagProposer(), _RegressingSolver(), t, good,
                          max_rounds=3, judge=lambda task, s: correct(task, s).ok)
    # started correct; a regressing rebuttal must be discarded by the guard
    assert correct(t, final).ok is True

def test_toy_mutator_makes_real_diff():
    from agent.genome import Genome, ToyMutator
    g0 = Genome()
    g1 = ToyMutator().mutate(g0)
    assert g1.revision == g0.revision + 1
    assert g1.to_json() != g0.to_json()   # a committable change exists


# --- [10] Phase 2: stricter paired stats + sigma inside the gate -------------

def test_t_based_paired_rejects_marginal_small_sample():
    # a tiny, noisy edge that the old pstdev+1.96 form could wave through; with
    # sample stdev + t (df=2) the lower bound should NOT clear 0.
    cand = [0.6, 0.55, 0.5]
    parent = [0.5, 0.5, 0.5]
    assert gate.paired_confirm(cand, parent) is False

def test_t_based_paired_accepts_strong_consistent_win():
    assert gate.paired_confirm([0.9, 0.92, 0.88, 0.91], [0.5, 0.5, 0.5, 0.5]) is True

def test_t95_monotonic_and_limits():
    assert gate._t95(1) > gate._t95(5) > gate._t95(30) > gate._t95(10**6)
    assert abs(gate._t95(10**6) - 1.645) < 1e-9

def test_sigma_gate_blocks_when_gain_below_noise():
    # a real but small mean gain (0.1) with sigma=0.1 -> margin 0.2 -> blocked
    cand = gate.Score(1.0, 0.0); parent = gate.Score(0.9, 0.0)
    blocked = gate.should_promote(
        cand, parent, candidate_dev=1.0, parent_dev=0.0, regression_ok=True,
        candidate_runs=[1.0, 1.0, 1.0], parent_runs=[0.9, 0.9, 0.9],
        anchor_ok=True, sigma=0.1, sigma_mult=2.0)
    assert blocked is False

def test_sigma_gate_allows_gain_above_noise():
    cand = gate.Score(1.0, 0.0); parent = gate.Score(0.0, 0.0)
    ok = gate.should_promote(
        cand, parent, candidate_dev=1.0, parent_dev=0.0, regression_ok=True,
        candidate_runs=[1.0, 1.0, 1.0, 1.0], parent_runs=[0.0, 0.0, 0.0, 0.0],
        anchor_ok=True, sigma=0.05, sigma_mult=2.0)
    assert ok is True


# --- [11] Phase 3: hardened sandbox -----------------------------------------

def test_verdict_from_fd_not_stdout_antispoof():
    # a verifier that prints a fake positive verdict to stdout must NOT fool us;
    # the real verdict comes from the result file.
    spoof = ('def check(s):\n'
             '    print(\'{"ok": true}\')\n'
             '    return False\n')
    assert sandbox.verify(spoof, "x")["ok"] is False

def test_env_is_stripped():
    # the child should not see arbitrary host env vars
    leak = ('def check(s):\n'
            '    import os\n'
            '    return "SECRET_TOKEN" in os.environ\n')
    import os as _os
    _os.environ["SECRET_TOKEN"] = "should-not-leak"
    try:
        assert sandbox.verify(leak, "x")["ok"] is False
    finally:
        _os.environ.pop("SECRET_TOKEN", None)

def test_correct_and_wrong_still_work_after_hardening():
    t = example_math.generate(seed=2)
    import re as _re
    truth = str(sum(int(x) for x in _re.findall(r"\d+", t.prompt)[:2]))
    assert verifier.judge(t, truth).ok is True
    assert verifier.judge(t, "-1").ok is False

def test_memory_bomb_is_bounded():
    # allocating far beyond the mem limit must fail (MemoryError) -> ok False,
    # not crash the parent. Keep the limit small for speed.
    bomb = 'def check(s):\n    x = bytearray(10**9)\n    return True\n'
    res = sandbox.verify(bomb, "x", timeout=5.0, mem_bytes=64 * 1024 * 1024)
    assert res["ok"] is False

def test_require_hardening_flag_present():
    # smoke: flag accepted; on POSIX CI it should still produce a verdict
    res = sandbox.verify("def check(s): return s=='ok'", "ok", require_hardening=True)
    assert res["ok"] is True


# --- [12] Phase 4: orchestrator integration ---------------------------------

def test_archive_softmax_sampling_is_diverse_not_greedy():
    from orchestrator.loop import Archive
    from agent.agent import Agent
    import random as _r
    arc = Archive()
    arc.add(Agent("hi"), 1.0)     # best
    arc.add(Agent("lo"), 0.0)     # worst
    rng = _r.Random(0)
    names = [arc.sample(rng).agent.name for _ in range(400)]
    hi, lo = names.count("hi"), names.count("lo")
    assert hi > lo                 # better is likelier (exploitation)
    assert lo > 0                  # but worst still sampled (exploration, non-greedy)

def test_measure_feeds_real_scores():
    from orchestrator.loop import measure
    from agent.agent import Agent
    tasks = [example_math.generate(seed=i) for i in range(5)]
    good = Agent("good")                       # toy solver is correct
    assert measure(good, tasks) == [1.0] * 5

def test_gate_promotes_a_genuine_measured_improvement():
    # weak parent (wrong) vs correct child, REAL measured paired runs -> promote
    from orchestrator.loop import measure
    from agent.agent import Agent
    class _Weak(Agent):
        def solve(self, task): return "-999"   # always wrong
    tasks = [example_math.generate(seed=i) for i in range(6)]
    parent_runs = measure(_Weak("weak"), tasks)
    child_runs = measure(Agent("ok"), tasks)
    assert parent_runs == [0.0] * 6 and child_runs == [1.0] * 6
    promoted = gate.should_promote(
        gate.Score(1.0, 0.0), gate.Score(0.0, 0.0),
        candidate_dev=1.0, parent_dev=0.0, regression_ok=True,
        candidate_runs=child_runs, parent_runs=parent_runs,
        anchor_ok=True, sigma=0.0)
    assert promoted is True

def test_apply_and_commit_refuses_empty_commit(tmp_path):
    import subprocess
    from orchestrator.loop import apply_and_commit
    from agent.genome import Genome
    repo = tmp_path
    subprocess.run(["git", "init", "-q"], cwd=repo)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=repo)
    subprocess.run(["git", "config", "user.name", "t"], cwd=repo)
    gpath = repo / "genome.json"
    g0 = Genome(revision=0)
    # first write is a real diff -> commits
    assert apply_and_commit(g0, "init", path=gpath, cwd=str(repo)) is True
    # same genome again -> NO diff -> must refuse (no fake empty commit)
    assert apply_and_commit(g0, "noop", path=gpath, cwd=str(repo)) is False
    # a real mutation -> diff again -> commits
    g1 = Genome(revision=1)
    assert apply_and_commit(g1, "promote", path=gpath, cwd=str(repo)) is True

def test_full_loop_runs_without_llm_or_git_sideeffects():
    from orchestrator import loop
    # smoke: completes, no exceptions, no commits (do_commit defaults False)
    loop.main(generations=4, seed=1, do_commit=False)
