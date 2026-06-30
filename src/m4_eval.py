from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math, re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", (text or "").lower()))


def _overlap_score(left: str, right: str) -> float:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _aggregate(per_question: list[EvalResult]) -> dict:
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    if not per_question:
        return {m: 0.0 for m in metrics}
    return {
        m: float(sum(getattr(r, m) for r in per_question) / len(per_question))
        for m in metrics
    }


def _heuristic_eval(
    questions: list[str],
    answers: list[str],
    contexts: list[list[str]],
    ground_truths: list[str],
) -> dict:
    per_question: list[EvalResult] = []
    for question, answer, ctxs, ground_truth in zip(questions, answers, contexts, ground_truths):
        context_text = " ".join(ctxs or [])
        per_question.append(
            EvalResult(
                question=question,
                answer=answer,
                contexts=ctxs,
                ground_truth=ground_truth,
                faithfulness=float(_overlap_score(answer, context_text)),
                answer_relevancy=float(_overlap_score(question, answer)),
                context_precision=float(max((_overlap_score(question, ctx) for ctx in ctxs), default=0.0)),
                context_recall=float(_overlap_score(ground_truth, context_text)),
            )
        )
    return {**_aggregate(per_question), "per_question": per_question}


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    # Implemented RAGAS evaluation.
    # 1. Wrap trong try/except — RAGAS cần OPENAI_API_KEY và Python 3.11+.
    # try:
    #     from ragas import evaluate
    #     from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    #     from datasets import Dataset
    #
    #     dataset = Dataset.from_dict({
    #         "question": questions, "answer": answers,
    #         "contexts": contexts, "ground_truth": ground_truths,
    #     })
    #     result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
    #                                         context_precision, context_recall])
    #     df = result.to_pandas()
    #     per_question = [EvalResult(question=row["question"], answer=row["answer"],
    #         contexts=row["contexts"], ground_truth=row["ground_truth"],
    #         faithfulness=float(row.get("faithfulness", 0.0)),
    #         answer_relevancy=float(row.get("answer_relevancy", 0.0)),
    #         context_precision=float(row.get("context_precision", 0.0)),
    #         context_recall=float(row.get("context_recall", 0.0)))
    #         for _, row in df.iterrows()]
    #     return {"faithfulness": ..., "answer_relevancy": ...,
    #             "context_precision": ..., "context_recall": ..., "per_question": [...]}
    # except Exception as e:
    #     print(f"  ⚠️  RAGAS evaluation failed: {e}")
    #     return zeros
    if not questions:
        return _heuristic_eval(questions, answers, contexts, ground_truths)

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        dataset = Dataset.from_dict(
            {
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            }
        )
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        df = result.to_pandas()

        def clean(value) -> float:
            try:
                number = float(value)
                return 0.0 if math.isnan(number) else number
            except Exception:
                return 0.0

        per_question = [
            EvalResult(
                question=str(row.get("question", "")),
                answer=str(row.get("answer", "")),
                contexts=list(row.get("contexts", []) or []),
                ground_truth=str(row.get("ground_truth", "")),
                faithfulness=clean(row.get("faithfulness", 0.0)),
                answer_relevancy=clean(row.get("answer_relevancy", 0.0)),
                context_precision=clean(row.get("context_precision", 0.0)),
                context_recall=clean(row.get("context_recall", 0.0)),
            )
            for _, row in df.iterrows()
        ]
        return {**_aggregate(per_question), "per_question": per_question}
    except Exception as e:
        print(f"  Warning: RAGAS evaluation failed, using heuristic fallback: {e}")
        return _heuristic_eval(questions, answers, contexts, ground_truths)


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    # Implemented failure analysis.
    # 1. diagnostic_tree = {
    #        "faithfulness": ("LLM hallucinating", "Tighten prompt, lower temperature"),
    #        "context_recall": ("Missing relevant chunks", "Improve chunking or add BM25"),
    #        "context_precision": ("Too many irrelevant chunks", "Add reranking or metadata filter"),
    #        "answer_relevancy": ("Answer doesn't match question", "Improve prompt template"),
    #    }
    # 2. For each EvalResult: compute avg of 4 metrics, find worst_metric
    # 3. Sort by avg ascending → take bottom_n
    # 4. Return [{"question": ..., "worst_metric": ..., "score": ...,
    #             "diagnosis": ..., "suggested_fix": ...}]
    diagnostic_tree = {
        "faithfulness": (
            "LLM answer is not well supported by retrieved context",
            "Tighten the answer prompt, lower temperature, and cite only retrieved evidence",
        ),
        "context_recall": (
            "Relevant evidence is missing from the retrieved contexts",
            "Improve chunking, add BM25/hybrid retrieval, or raise retrieval top_k",
        ),
        "context_precision": (
            "Retrieved contexts contain too much irrelevant material",
            "Add reranking, metadata filters, or smaller child chunks",
        ),
        "answer_relevancy": (
            "Answer does not directly address the question",
            "Improve the prompt template and pass the question more explicitly to generation",
        ),
    }

    scored: list[tuple[float, EvalResult, str]] = []
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for result in eval_results:
        metric_values = {name: float(getattr(result, name, 0.0)) for name in metric_names}
        avg_score = sum(metric_values.values()) / len(metric_values)
        worst_metric = min(metric_values, key=metric_values.get)
        scored.append((avg_score, result, worst_metric))

    scored.sort(key=lambda item: item[0])
    failures: list[dict] = []
    for avg_score, result, worst_metric in scored[:bottom_n]:
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        failures.append(
            {
                "question": result.question,
                "answer": result.answer,
                "ground_truth": result.ground_truth,
                "worst_metric": worst_metric,
                "score": float(getattr(result, worst_metric, 0.0)),
                "average_score": float(avg_score),
                "diagnosis": diagnosis,
                "suggested_fix": suggested_fix,
            }
        )
    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
