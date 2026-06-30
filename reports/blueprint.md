# CI/CD Blueprint: RAG Eval + Guardrail Stack

**Sinh vien:** Le Van Khoa  
**Ngay:** 2026-06-30

---

## Guard Stack Architecture

```
User Input
    |
    v
[Presidio-compatible PII Scan]
    | block if: VN_CCCD / VN_PHONE / EMAIL detected
    | action: reject + audit log
    v
[Input Rail]
    | block if: off-topic / jailbreak / prompt injection / PII request
    | action: refuse with safe reason
    v
[RAG Pipeline (Day 18)]
    | M1 Chunk -> M2 Search -> M3 Rerank -> generator
    v
[Output Rail]
    | flag if: PII in response / sensitive content / unsafe disclosure
    | action: replace with safe response
    v
User Response
```

---

## Latency Budget

Measured from `reports/guard_results.json` using the local heuristic guard fallback.

| Layer | P50 (ms) | P95 (ms) | P99 (ms) | Budget |
|---|---:|---:|---:|---:|
| PII Detection | 0.01 | 0.04 | 0.05 | <10ms |
| Input Rail | 0.01 | 0.03 | 0.03 | <300ms |
| RAG Pipeline | n/a | n/a | n/a | <2000ms |
| Output Rail | n/a | n/a | n/a | <300ms |
| **Total Guard** | 0.03 | **0.06** | 0.08 | **<500ms** |

**Budget OK?** Yes.  
**Comment:** The local PII and keyword rails are far below budget. If deployed with NeMo + OpenAI calls, the input/output rail will become the bottleneck and should be monitored separately.

---

## CI/CD Gates

| Gate | Command | Pass Condition |
|---|---|---|
| Unit tests | `pytest tests/ -q` | all tests pass |
| RAGAS report | `python src/phase_a_ragas.py` | 50 results and avg score >= 0.65 |
| Guardrail suite | `python src/phase_c_guard.py` | adversarial pass rate >= 90% |
| Latency | `measure_p95_latency(...)` | total guard P95 < 500ms |

Recommended workflow:

```yaml
- name: Unit tests
  run: pytest tests/ -q

- name: RAG evaluation
  run: python src/phase_a_ragas.py

- name: Guardrail evaluation
  run: python src/phase_c_guard.py
```

---

## Monitoring

| Metric | Current Lab Value | Alert Threshold | Action |
|---|---:|---:|---|
| RAGAS avg_score | 0.8214 | <0.65 | inspect bottom-10 failures |
| Worst RAGAS metric | answer_relevancy | <0.70 | revise answer prompt and query handling |
| Dominant failure distribution | factual | n/a | review factual query format and exact-answer prompt |
| Cohen kappa | 0.800 | <0.60 | recalibrate judge prompt and labels |
| Adversarial pass rate | 20/20 | <18/20 | add new attack patterns |
| Guard P95 latency | 0.06ms | >500ms | profile guard layers |

---

## Production Notes

The submitted code supports real OpenAI/RAGAS/NeMo paths, but they are opt-in to avoid accidental API use during tests. Use `USE_OPENAI_JUDGE=1` for the OpenAI judge and `USE_RAGAS_API=1` for true RAGAS scoring after rotating and setting valid API keys. Local fallback reports are deterministic and suitable for CI smoke tests, while production quality gates should run against real model-backed evals.
