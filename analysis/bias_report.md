# LLM Judge Bias Report - Phase B

**Sinh vien:** Le Van Khoa  
**Ngay:** 2026-06-30  
**Judge model:** gpt-4o-mini path available; local heuristic fallback used by default

---

## 1. Pairwise Judge Results

The report was generated at `reports/judge_results.json`. The local judge compared each model answer against a weak fallback answer for the first 5 labeled examples.

| # | Question summary | Winner | Reasoning summary |
|---:|---|---|---|
| 1 | marriage leave | A | A is more relevant and policy-specific |
| 2 | 55M equipment approval | A | A is more relevant than the weak fallback |
| 3 | Tet bonus minimum | A | A directly answers with the bonus amount |
| 4 | Senior tenure leave and salary | A | A includes both requested facts |
| 5 | training reimbursement | A | A answers the repayment amount |

---

## 2. Swap-and-Average Results

| # | Pass 1 Winner | Pass 2 Winner | Final | Position Consistent? |
|---:|---|---|---|---|
| 1 | A | A | A | Yes |
| 2 | A | A | A | Yes |
| 3 | A | A | A | Yes |
| 4 | A | A | A | Yes |
| 5 | A | A | A | Yes |

**Position bias rate:** 0.0% (0/5 inconsistent).

---

## 3. Cohen Kappa Analysis

Human labels were loaded from `human_labels_10q.json`. Judge labels were produced by the local fallback scorer for the same 10 examples.

| # | Human Label | Judge Label | Agree? |
|---:|---:|---:|---|
| 1 | 1 | 1 | Yes |
| 2 | 0 | 0 | Yes |
| 3 | 1 | 1 | Yes |
| 4 | 1 | 1 | Yes |
| 5 | 1 | 0 | No |
| 6 | 0 | 0 | Yes |
| 7 | 1 | 1 | Yes |
| 8 | 0 | 0 | Yes |
| 9 | 1 | 1 | Yes |
| 10 | 0 | 0 | Yes |

**Cohen kappa:** 0.800  
**Interpretation:** almost perfect / substantial agreement for this small labeled set.

---

## 4. Verbosity Bias

For decisive cases:

- A wins and A is longer than B: 2 / 5 cases
- B wins and B is longer than A: 0 / 5 cases
- **Verbosity bias rate:** 40.0%

The judge did not show severe verbosity bias on this sample. The rate is still worth monitoring because the comparison used a short weak fallback answer, so longer correct model answers naturally win.

---

## 5. Notes

The OpenAI judge path is implemented but opt-in through `USE_OPENAI_JUDGE=1` to avoid accidental API usage in unit tests. For production, use the real judge with JSON mode, keep swap-and-average enabled, and periodically compare against human labels. If kappa drops below 0.6, refresh the rubric prompt and add more labeled edge cases.
