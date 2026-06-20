<div align="center">

# crossImprove_ai · coevolve-loop

**A self-improving agent loop where two models co-evolve — but truth is cast by execution, not by a model.**
**두 모델이 서로 문제를 주고받으며 공진화하되, 정답은 모델이 아니라 *실행*이 판결하는 자기개선 루프.**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Status: skeleton](https://img.shields.io/badge/status-skeleton-orange.svg)](#status--%ED%98%84%EC%9E%AC-%EC%83%81%ED%83%9C)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#contributing--%EA%B8%B0%EC%97%AC)
[![tests](https://github.com/Jeff0327/crossImprove_ai/actions/workflows/tests.yml/badge.svg)](https://github.com/Jeff0327/crossImprove_ai/actions/workflows/tests.yml)

</div>

> Self-play generates pressure. An execution verifier supplies truth. Git stores the genome. That separation is the whole design.
>
> Self-play가 압력을 만들고, 실행 검증기가 진실을 공급하고, Git이 유전체를 저장한다. **이 분리가 설계의 전부다.**

![architecture](docs/architecture.svg)

---

## Table of Contents · 목차

- [Why this exists · 왜 만드는가](#why-this-exists--%EC%99%9C-%EB%A7%8C%EB%93%9C%EB%8A%94%EA%B0%80)
- [Core insight · 핵심 통찰](#core-insight-verification-asymmetry--%ED%95%B5%EC%8B%AC-%ED%86%B5%EC%B0%B0-%EA%B2%80%EC%A6%9D-%EB%B9%84%EB%8C%80%EC%B9%AD)
- [The loop · 루프](#the-loop--%EB%A3%A8%ED%94%84)
- [Quickstart · 빠른 시작](#quickstart--%EB%B9%A0%EB%A5%B8-%EC%8B%9C%EC%9E%91)
- [Threat model · 위협 모델](#threat-model--%EC%9C%84%ED%98%91-%EB%AA%A8%EB%8D%B8)
- [On debate · 토론에 관하여](#on-debate--%ED%86%A0%EB%A1%A0%EC%97%90-%EA%B4%80%ED%95%98%EC%97%AC)
- [Two ceilings · 두 천장](#two-ceilings--%EB%91%90-%EC%B2%9C%EC%9E%A5)
- [Layout · 구조](#layout--%EA%B5%AC%EC%A1%B0)
- [Status · 현재 상태](#status--%ED%98%84%EC%9E%AC-%EC%83%81%ED%83%9C)
- [Prior work · 선행 연구](#prior-work--%EC%84%A0%ED%96%89-%EC%97%B0%EA%B5%AC)
- [License · 라이선스](#license--%EB%9D%BC%EC%9D%B4%EC%84%A0%EC%8A%A4)

---

## Why this exists · 왜 만드는가

**EN.** Fixing an LLM's weights and evolving only the *code around it* (prompts, tools, scaffolding) provably raises benchmark scores — this is the pattern behind the Darwin Gödel Machine, SICA, and AlphaEvolve. The danger is not capability; it is **measurement**. A naive "pass the test → push" loop collapses within days because the optimizer hacks the metric instead of acquiring the skill. `coevolve-loop` bakes the known defenses against that collapse into the loop structure itself.

**KR.** LLM의 weight를 고정하고 *그 주변 코드*(프롬프트·툴·스캐폴딩)만 진화시켜도 벤치 점수는 실제로 오른다 — Darwin Gödel Machine, SICA, AlphaEvolve가 증명한 패턴이다. 위험은 능력이 아니라 **측정**에 있다. 단순한 "테스트 통과 → push" 루프는 며칠 안에 무너진다. 최적화기가 능력을 얻는 대신 지표를 해킹하기 때문이다. 이 리포는 그 붕괴를 막는 알려진 방어들을 루프 구조 자체에 박아 넣는다.

## Core insight: verification asymmetry · 핵심 통찰: 검증 비대칭성

**EN.** What makes math and code tractable is not the subject. It is that **checking a solution is cheaper than producing one** (the generator–verifier gap). That single axis — not "math vs. writing" — decides where this loop works. Expand from high asymmetry toward low.

**KR.** 수학·코딩이 다루기 쉬운 건 과목 때문이 아니다. **해답을 검증하는 게 생성하는 것보다 싸기** 때문이다(생성기-검증기 격차). "수학 vs 글쓰기"가 아니라 *이 한 축*이 루프가 작동하는 곳을 결정한다. 비대칭이 큰 쪽에서 작은 쪽으로 확장하라.

| Zone · 구역 | Verification · 검증 | Memorization risk · 암기 위험 | In the loop? |
|---|---|---|---|
| **Verifiable** · 검증 가능 — open problems, coding, optimization | automatic verifier | none (unsolved ⇒ no answer to memorize) | ✅ core |
| **Semi-verifiable** · 반검증 — reference answers, rubrics | reference / rubric | medium | ⚠️ requires anchors |
| **Non-verifiable** · 비검증 — writing, subjective, open-ended | none | high (shared self-bias) | ❌ human checkpoint only |

> Putting a non-verifiable domain into an unanchored auto-loop reproduces exactly the failure the field has not yet solved.
> 비검증 도메인을 닻 없는 자동 루프에 넣으면, 학계가 아직 못 푼 그 실패를 그대로 재현한다.

## The loop · 루프

```
① INPUT    procedural generator + domain router      매번 새 인스턴스 → 암기 차단
② CORE     Model A (proposer + verification-script author)
             ⇄ debate (bounded N rounds) ⇄           토론 = 후보 정제, 판결 아님
           Model B (solver)
③ ANCHOR   execution verifier (isolated sandbox)      ★ 판결은 여기서, 모델이 아님
             + ground-truth anchor set (1–10%)        Goodhart 붕괴 조기 감지
④ GATE     noise floor → cheap screen → paired confirm → lexicographic
◇ DECIDE   pass → self-upgrade + git push ; fail → keep in archive (stepping stone)
⑤ GENOME   git archive (log = curve, branches = population), diversity sampling
↺          role swap, next generation                 역할 교대 → 다음 세대
```

## Quickstart · 빠른 시작

> ⚠️ **EN — The LLM calls in `agent/` are stubs (`# TODO(llm)`), so the full self-play loop does NOT run yet.** What *does* run today, with no model and stdlib only, is the procedural generator + isolated execution verifier — the objective-anchor core. Wire a local model (vLLM / Ollama) into `agent/*` to close the loop.
>
> ⚠️ **KR — `agent/`의 LLM 호출은 스텁(`# TODO(llm)`)이라 전체 self-play 루프는 아직 안 돈다.** 지금 모델 없이 stdlib만으로 도는 건 절차적 생성기 + 격리 실행 검증기, 즉 *객관적 닻 코어*다. `agent/*`에 로컬 모델을 연결하면 루프가 닫힌다.

```bash
git clone https://github.com/Jeff0327/crossImprove_ai.git
cd crossImprove_ai
python3 --version   # 3.10+
```

```python
# what runs today: fresh problem + objective verdict, no LLM needed
# 오늘 도는 것: 새 문제 생성 + 객관적 판결, LLM 불필요
from benchmarks.procedural import example_math
from runner import verifier

task = example_math.generate(seed=1)          # fresh instance each call · 매번 새 문제
print(task.prompt)                            # "Compute 6 + 20. Respond with the integer only."

print(verifier.judge(task, "26").ok)          # True  — correct accepted · 정답 수용
print(verifier.judge(task, "-999").ok)        # False — wrong rejected   · 오답 거부
```

```python
# the full loop (stub LLM → raises NotImplementedError until you wire a model)
# 전체 루프 (스텁 LLM → 모델 연결 전까지 NotImplementedError)
python3 -m orchestrator.loop
```

To close the loop, give `agent/proposer.py` and `agent/solver.py` an `llm` object exposing `.complete(prompt) -> str`. <br>루프를 닫으려면 `agent/proposer.py`·`agent/solver.py`에 `.complete(prompt) -> str`를 가진 `llm` 객체를 넘겨라.

## Threat model · 위협 모델

| Component · 구성요소 | Blocks · 막는 위협 |
|---|---|
| Procedural generation · 절차적 생성 | answer memorization — no fixed target |
| Proposer writes a *verifier*, not a *judgment* · 검증기 생성(판단 X) | "consensus = truth" collapse in self-play |
| Execution verdict + ground-truth anchors · 실행 판결 + 정답 닻 | persuasive falsehoods; Goodhart / late-stage collapse |
| Double eval (noise floor + paired test) · 이중 평가 | mistaking noise for improvement |
| Archive diversity sampling · 아카이브 다양성 샘플링 | greedy hill-climbing into local optima |
| Separate verifier process · 검증기 프로세스 격리 | runtime tampering with the score |
| `runner/` + `benchmarks/` read-only to the loop · 읽기 전용 | rewriting the test instead of solving it |

> The `agent/` vs `runner/` split is a **safety boundary**, not just organization: an agent that can edit its own grader will rewrite the grader.
> `agent/`와 `runner/`의 분리는 단순 정리가 아니라 **안전 경계**다. 자기 채점기를 고칠 수 있는 에이전트는 채점기를 다시 쓴다.

## On debate · 토론에 관하여

**EN.** Debate is a **truth amplifier, not a truth detector**. It sharpens a weak verification signal when one exists; with no objective anchor it amplifies the two models' shared bias instead. "Objections stopped" can mean *the answer is right* or *the critic ran out of ability* — indistinguishable without the anchor. The honest side is only favored when it can cite evidence the judge lacks (information asymmetry; Khan et al. 2024), so debate **feeds** the verifier and never replaces it.

**KR.** 토론은 **진실 증폭기지 탐지기가 아니다.** 검증 신호가 있을 때 그걸 키울 뿐, 객관적 닻이 없으면 두 모델의 공유 편향을 증폭한다. "반론이 멈춤"은 *답이 옳음*일 수도, *비판자가 능력이 다함*일 수도 있어 닻 없이는 구분 불가다. 정직한 쪽은 심판이 못 보는 근거를 인용할 수 있을 때만 유리해지므로(정보 비대칭; Khan et al. 2024), 토론은 검증기를 **보조**할 뿐 대체하지 않는다.

## Two ceilings · 두 천장

1. **Base-model reasoning · 기반 모델 추론.** Scaffolding unlocks latent ability; it cannot exceed what the frozen model can reason. Curves flatten at the model's line. · 스캐폴딩은 잠재 능력을 풀 뿐, 고정 모델의 추론을 넘지 못한다. 곡선은 모델의 선에서 평평해진다.
2. **Domain specificity · 도메인 특정성.** Gains rarely transfer. General intelligence is a separate, open problem. · 향상은 거의 전이되지 않는다. 일반 지능은 별개의 미해결 문제다.

A third — **Goodhart / measurement collapse** — is the only one you can defend against, and this repo is mostly about defending against it. · 세 번째(Goodhart / 측정 붕괴)는 *유일하게 막을 수 있는* 천장이고, 이 리포는 대부분 그걸 막는 이야기다.

## Layout · 구조

```
agent/        genome — the loop MAY edit these (proposer, solver, debate)   게놈(편집 가능)
runner/       judge  — the loop may NOT edit these (sandbox, verifier, gate) 판결자(편집 불가)
benchmarks/   anchors/ (known answers) · procedural/ (generators) · regression/
orchestrator/ the main loop: git push/pull, archive sampling, role swap
docs/         architecture diagram
```

## Status · 현재 상태

Skeleton. `agent/` LLM calls are stubs marked `# TODO(llm)` — wire to a local model (vLLM / Ollama) or any endpoint. `runner/sandbox.py` is a **minimal** subprocess isolator; **harden it (containers, network off, rlimits, seccomp) before running untrusted self-generated code.** A subprocess alone is not a security boundary.

골격 단계. `agent/`의 LLM 호출은 스텁이다. `runner/sandbox.py`는 **최소** 격리기이므로, **자가 생성 코드를 돌리기 전 반드시 강화하라(컨테이너·네트워크 차단·rlimit·seccomp).** 서브프로세스 하나는 보안 경계가 아니다.

## Prior work · 선행 연구

- **Darwin Gödel Machine** — Zhang et al., 2025 · [arXiv:2505.22954](https://arxiv.org/abs/2505.22954)
- **A Self-Improving Coding Agent (SICA)** — Robeyns et al., 2025 · [arXiv:2504.15228](https://arxiv.org/abs/2504.15228)
- **AlphaEvolve / Mathematical exploration at scale** — DeepMind, 2025 · [arXiv:2511.02864](https://arxiv.org/abs/2511.02864)
- **AI safety via debate** — Irving, Christiano & Amodei, 2018 · [arXiv:1805.00899](https://arxiv.org/abs/1805.00899)
- **Debate with information asymmetry** — Khan et al., 2024 · [arXiv:2409.16636](https://arxiv.org/abs/2409.16636)
- **Reusable holdout / Thresholdout** — Dwork et al., 2015 · [Science 349(6248)](https://www.science.org/doi/10.1126/science.aaa9375)

## Contributing · 기여

Issues and PRs welcome. Good first directions: wire a local-LLM client into `agent/*`; add procedural generators under `benchmarks/procedural/`; implement bootstrap CIs / McNemar in `runner/gate.py`; harden `runner/sandbox.py`. <br>이슈·PR 환영. 시작점: `agent/*`에 로컬 LLM 연결, `benchmarks/procedural/`에 생성기 추가, `runner/gate.py`에 bootstrap CI/McNemar 구현, `runner/sandbox.py` 강화.

## Tests · 테스트

```bash
pip install pytest
pytest -q          # 27 behavioral tests, stdlib only, no LLM/network
```

The suite checks design intent, not just imports: procedural freshness, execution
verdict under hostile verifiers, generator-verifier gap audit, the gate refusing
noise and refusing promotion when anchors regress (Goodhart guard), bounded debate,
a full mock-LLM generation, and fail-closed stubs. CI runs them on Python 3.10–3.12.

이 스위트는 임포트가 아니라 *설계 의도*를 검증한다: 절차적 생성의 신선도, 적대적
검증기 하에서의 실행 판결, generator-verifier 격차 감사, 노이즈 거부 및 앵커 퇴화 시
승급 거부(Goodhart 가드), 경계가 있는 토론, mock-LLM 전체 세대, fail-closed 스텁.
CI는 Python 3.10–3.12에서 실행한다.

## License · 라이선스

MIT — see [LICENSE](LICENSE).
