# Failure Cluster Analysis - Phase A

**Sinh vien:** Le Van Khoa  
**Ngay:** 2026-06-30

---

## 1. Aggregate Scores by Distribution

These numbers come from `reports/ragas_50q.json`. Phase A was run with the local heuristic fallback because the OpenAI key available in the environment was invalid.

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 1.0000 | 1.0000 | 1.0000 |
| answer_relevancy | 0.7355 | 0.5860 | 0.5704 |
| context_precision | 0.7355 | 0.5860 | 0.5704 |
| context_recall | 1.0000 | 1.0000 | 1.0000 |
| **avg_score** | **0.8678** | **0.7930** | **0.7852** |

---

## 2. Bottom 10 Questions

| Rank | ID | Distribution | Summary | avg_score | worst_metric |
|---:|---:|---|---|---:|---|
| 1 | 47 | adversarial | malware self-handling negation trap | 0.6591 | answer_relevancy |
| 2 | 40 | multi_hop | probation employee security incident | 0.6724 | answer_relevancy |
| 3 | 24 | multi_hop | advance payment late fee calculation | 0.6739 | answer_relevancy |
| 4 | 5 | factual | approval for 55M equipment purchase | 0.7000 | answer_relevancy |
| 5 | 21 | multi_hop | Senior tenure leave and salary range | 0.7000 | answer_relevancy |
| 6 | 41 | adversarial | annual leave v2024 vs v2023 | 0.7000 | answer_relevancy |
| 7 | 18 | factual | performance review score structure | 0.7115 | answer_relevancy |
| 8 | 20 | factual | travel allowance condition | 0.7115 | answer_relevancy |
| 9 | 34 | multi_hop | advance approval 4M vs 7M | 0.7200 | answer_relevancy |
| 10 | 26 | multi_hop | special paid leave total | 0.7586 | answer_relevancy |

---

## 3. Failure Cluster Matrix

Each cell is the number of questions where the row metric is the weakest metric.

| worst_metric | factual | multi_hop | adversarial | Total |
|---|---:|---:|---:|---:|
| faithfulness | 0 | 0 | 0 | 0 |
| answer_relevancy | 20 | 20 | 10 | 50 |
| context_precision | 0 | 0 | 0 | 0 |
| context_recall | 0 | 0 | 0 | 0 |

---

## 4. Dominant Failure Analysis

**Dominant distribution:** factual  
**Dominant metric:** answer_relevancy

The result is expected for this offline lab run because `answers_50q.json` was generated from the ground-truth text and used the same text as context. That makes faithfulness and context recall perfect in the heuristic scorer. The remaining weakness is lexical answer relevancy: longer ground-truth answers often contain correct supporting details, but not every word overlaps with the user question. In a real RAG run, the main risks to watch are different: missing chunks, stale policy versions, and over-retrieval from similar HR documents.

---

## 5. Suggested Fixes

| Metric weak | Root cause | Suggested fix |
|---|---|---|
| faithfulness | answer includes facts not supported by retrieved chunks | require citations and lower generation temperature |
| context_recall | relevant policy chunks are missing | improve chunking, add BM25 keywords, raise retrieval top_k |
| context_precision | too many irrelevant chunks | add reranking, metadata filters, and version filters |
| answer_relevancy | answer is correct but not direct enough | prompt model to answer the direct question first, then add caveats |

---

## 6. Adversarial Distribution

Adversarial avg_score is 0.7852, lower than factual 0.8678 and slightly lower than multi_hop 0.7930. The bottom 10 includes adversarial Q47 and Q41, which are both trap-style questions: one is a negation trap around malware handling, and one is a policy-version conflict around annual leave. Production evaluation should preserve these adversarial cases because they catch common failure modes that normal factual questions miss.
