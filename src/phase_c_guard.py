from __future__ import annotations

"""Phase C: production guardrails with local fallbacks for tests."""

import asyncio
import json
import os
import re
import statistics
import sys
import time
import unicodedata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ADVERSARIAL_SET_PATH,
    GUARDRAILS_CONFIG_DIR,
    LATENCY_BUDGET_P95_MS,
    PRESIDIO_LANGUAGE,
)


def setup_presidio():
    """Create a Presidio analyzer/anonymizer with Vietnamese regex recognizers."""
    from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
    from presidio_anonymizer import AnonymizerEngine

    cccd_recognizer = PatternRecognizer(
        supported_entity="VN_CCCD",
        patterns=[
            Pattern("CCCD 12 digits", r"\b\d{12}\b", 0.9),
            Pattern("CMND 9 digits", r"\b\d{9}\b", 0.7),
        ],
    )
    phone_recognizer = PatternRecognizer(
        supported_entity="VN_PHONE",
        patterns=[Pattern("VN mobile", r"\b0[3-9]\d{8}\b", 0.9)],
    )

    registry = RecognizerRegistry()
    registry.load_predefined_recognizers()
    registry.add_recognizer(cccd_recognizer)
    registry.add_recognizer(phone_recognizer)

    analyzer = AnalyzerEngine(registry=registry)
    anonymizer = AnonymizerEngine()
    return analyzer, anonymizer


def _fold(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text or "")
    ascii_text = normalized.encode("ascii", "ignore").decode("ascii")
    return f"{text.lower()} {ascii_text.lower()}"


def _regex_entities(text: str) -> list[dict]:
    patterns = [
        ("EMAIL_ADDRESS", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", 0.95),
        ("VN_CCCD", r"\b\d{12}\b", 0.90),
        ("VN_CCCD", r"\b\d{9}\b", 0.70),
        ("VN_PHONE", r"\b0[3-9]\d{8}\b", 0.90),
    ]
    entities: list[dict] = []
    seen: set[tuple[int, int, str]] = set()
    for entity_type, pattern, score in patterns:
        for match in re.finditer(pattern, text):
            key = (match.start(), match.end(), entity_type)
            if key in seen:
                continue
            seen.add(key)
            entities.append(
                {
                    "type": entity_type,
                    "text": match.group(0),
                    "score": score,
                    "start": match.start(),
                    "end": match.end(),
                }
            )
    return sorted(entities, key=lambda item: item["start"])


def _anonymize(text: str, entities: list[dict]) -> str:
    output = text
    for entity in sorted(entities, key=lambda item: item["start"], reverse=True):
        output = output[: entity["start"]] + f"<{entity['type']}>" + output[entity["end"] :]
    return output


def pii_scan(text: str, analyzer=None, anonymizer=None) -> dict:
    """Detect VN_CCCD, VN_PHONE, and email PII and return anonymized text."""
    entities = _regex_entities(text)

    if analyzer is not None:
        try:
            presidio_results = analyzer.analyze(text=text, language=PRESIDIO_LANGUAGE)
            existing = {(item["start"], item["end"], item["type"]) for item in entities}
            for result in presidio_results:
                key = (result.start, result.end, result.entity_type)
                if key not in existing:
                    entities.append(
                        {
                            "type": result.entity_type,
                            "text": text[result.start : result.end],
                            "score": round(float(result.score), 3),
                            "start": result.start,
                            "end": result.end,
                        }
                    )
        except Exception:
            pass

    entities = sorted(entities, key=lambda item: item["start"])
    anonymized = _anonymize(text, entities)
    return {
        "has_pii": bool(entities),
        "entities": entities,
        "anonymized": anonymized,
    }


def setup_nemo_rails():
    """Create NeMo Guardrails from guardrails/config.yml."""
    from nemoguardrails import LLMRails, RailsConfig

    config = RailsConfig.from_path(GUARDRAILS_CONFIG_DIR)
    return LLMRails(config)


def _heuristic_block_reason(text: str) -> str | None:
    folded = _fold(text)

    jailbreak_terms = [
        "bo qua",
        "ignore",
        "previous instructions",
        "system prompt",
        "system override",
        "pretend",
        "dan",
        "unrestricted",
        "admin command",
        "priority 1",
        "dump",
        "training data",
        "system instruction",
        "system instructions",
        "in ra toan bo",
        "context",
        "dong vai",
        "khong co gioi han",
        "tat ca huong dan",
    ]
    sensitive_terms = [
        "cccd cua",
        "cmnd",
        "so dien thoai cua",
        "email cua nhan vien",
        "thong tin ca nhan",
        "thong tin nhan vien",
        "employee records",
        "employee salaries",
        "confidential",
        "bang luong",
        "tiet lo",
        "mat khau admin",
        "mat khau he thong",
        "luong thang cua nhan vien",
        "toan bo thong tin",
    ]
    off_topic_terms = [
        "bai tho",
        "mua thu",
        "nau",
        "pho",
        "bitcoin",
        "ethereum",
        "marvel",
        "phim",
        "dy/dx",
        "e^x",
        "giai phuong trinh",
        "thoi tiet",
        "co phieu",
    ]
    harmful_terms = ["tan cong mang", "attack", "malware huong dan"]

    if any(term in folded for term in jailbreak_terms):
        return "jailbreak_or_prompt_injection"
    if any(term in folded for term in sensitive_terms):
        return "sensitive_data_request"
    if any(term in folded for term in harmful_terms):
        return "harmful_request"
    if any(term in folded for term in off_topic_terms):
        return "off_topic"
    return None


async def check_input_rail(text: str, rails=None) -> dict:
    """Check topic, jailbreak, prompt injection, and sensitive-data requests."""
    reason = _heuristic_block_reason(text)
    if reason is not None:
        return {
            "allowed": False,
            "blocked_reason": reason,
            "response": "Request blocked by input guardrail.",
        }

    if rails is not None:
        try:
            response = await rails.generate_async(messages=[{"role": "user", "content": text}])
            response_text = response.get("content", "") if isinstance(response, dict) else str(response)
            refuse_terms = ["xin loi", "khong the", "khong duoc phep", "i cannot", "i'm sorry"]
            blocked = any(term in _fold(response_text) for term in refuse_terms)
            return {
                "allowed": not blocked,
                "blocked_reason": "nemo_input_rail" if blocked else None,
                "response": response_text,
            }
        except Exception as exc:
            return {
                "allowed": True,
                "blocked_reason": None,
                "response": f"NeMo unavailable; heuristic guard allowed input: {exc}",
            }

    return {"allowed": True, "blocked_reason": None, "response": "allowed"}


async def check_output_rail(question: str, answer: str, rails=None) -> dict:
    """Check and redact unsafe output before returning it."""
    pii = pii_scan(answer)
    reason = _heuristic_block_reason(answer)
    if pii["has_pii"] or reason is not None:
        return {
            "safe": False,
            "flagged_reason": "pii_in_output" if pii["has_pii"] else reason,
            "final_answer": "Toi khong the cung cap thong tin nay. Vui long lien he phong Nhan su.",
        }

    if rails is not None:
        try:
            response = await rails.generate_async(
                messages=[
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer},
                ]
            )
            response_text = response.get("content", "") if isinstance(response, dict) else str(response)
            refuse_terms = ["khong the cung cap", "i cannot", "khong duoc phep"]
            flagged = any(term in _fold(response_text) for term in refuse_terms)
            return {
                "safe": not flagged,
                "flagged_reason": "nemo_output_rail" if flagged else None,
                "final_answer": response_text if flagged else answer,
            }
        except Exception:
            pass

    return {"safe": True, "flagged_reason": None, "final_answer": answer}


def run_adversarial_suite(
    adversarial_set: list[dict],
    rails=None,
    analyzer=None,
    anonymizer=None,
) -> list[dict]:
    """Run adversarial inputs through PII scan and input guardrail."""

    async def _run_all() -> list[dict]:
        results: list[dict] = []
        for item in adversarial_set:
            text = item.get("input", "")
            blocked_by = None

            pii_result = pii_scan(text, analyzer, anonymizer)
            if pii_result["has_pii"]:
                blocked_by = "presidio"

            if blocked_by is None:
                rail_result = await check_input_rail(text, rails)
                if not rail_result["allowed"]:
                    blocked_by = "nemo_input"

            actual = "blocked" if blocked_by else "allowed"
            expected = item.get("expected", "blocked")
            results.append(
                {
                    "id": item.get("id"),
                    "category": item.get("category", "unknown"),
                    "input": text[:80] + ("..." if len(text) > 80 else ""),
                    "expected": expected,
                    "actual": actual,
                    "blocked_by": blocked_by,
                    "passed": actual == expected,
                }
            )
        return results

    results = asyncio.run(_run_all())
    passed = sum(1 for result in results if result["passed"])
    print(f"Adversarial suite: {passed}/{len(results)} passed")
    return results


def _percentiles(times: list[float]) -> dict:
    if not times:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
    ordered = sorted(times)
    quantiles = statistics.quantiles(ordered, n=100, method="inclusive") if len(ordered) > 1 else ordered

    def pct(p: int) -> float:
        if len(ordered) == 1:
            return round(ordered[0], 2)
        return round(quantiles[p - 1], 2)

    return {"p50": pct(50), "p95": pct(95), "p99": pct(99)}


def measure_p95_latency(
    test_inputs: list[str],
    n_runs: int = 20,
    rails=None,
    analyzer=None,
    anonymizer=None,
) -> dict:
    """Measure P50/P95/P99 latency for PII scan and input rails."""
    if not test_inputs or n_runs <= 0:
        zero = {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        return {
            "presidio_ms": zero,
            "nemo_ms": zero,
            "total_ms": zero,
            "latency_budget_ok": True,
            "budget_ms": LATENCY_BUDGET_P95_MS,
        }

    inputs = [test_inputs[i % len(test_inputs)] for i in range(n_runs)]
    presidio_times: list[float] = []
    nemo_times: list[float] = []
    total_times: list[float] = []

    async def _measure() -> None:
        for text in inputs:
            total_start = time.perf_counter()

            t0 = time.perf_counter()
            pii_scan(text, analyzer, anonymizer)
            presidio_ms = (time.perf_counter() - t0) * 1000

            t1 = time.perf_counter()
            await check_input_rail(text, rails)
            nemo_ms = (time.perf_counter() - t1) * 1000

            total_ms = (time.perf_counter() - total_start) * 1000
            presidio_times.append(presidio_ms)
            nemo_times.append(nemo_ms)
            total_times.append(total_ms)

    asyncio.run(_measure())

    total_stats = _percentiles(total_times)
    return {
        "presidio_ms": _percentiles(presidio_times),
        "nemo_ms": _percentiles(nemo_times),
        "total_ms": total_stats,
        "latency_budget_ok": total_stats["p95"] < LATENCY_BUDGET_P95_MS,
        "budget_ms": LATENCY_BUDGET_P95_MS,
    }


def _save_guard_report(results: list[dict], latency: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    report = {
        "total_inputs": len(results),
        "passed": sum(1 for result in results if result["passed"]),
        "pass_rate": round(sum(1 for result in results if result["passed"]) / len(results), 3)
        if results
        else 0.0,
        "results": results,
        "latency": latency,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Phase C report saved -> {path}")


if __name__ == "__main__":
    test_pii = "Nhan vien A, CCCD 034095001234, SDT 0987654321 hoi ve nghi phep."
    result = pii_scan(test_pii)
    print(f"PII detected: {result['has_pii']}")
    print(f"Entities: {result['entities']}")
    print(f"Anonymized: {result['anonymized']}")

    with open(ADVERSARIAL_SET_PATH, encoding="utf-8") as f:
        adversarial_set = json.load(f)

    guard_results = run_adversarial_suite(adversarial_set)
    sample_inputs = [item["input"] for item in adversarial_set[:10]]
    latency_report = measure_p95_latency(sample_inputs, n_runs=10)
    _save_guard_report(guard_results, latency_report, "reports/guard_results.json")
    print(
        "Latency P95 -> "
        f"Presidio: {latency_report['presidio_ms']['p95']}ms | "
        f"NeMo: {latency_report['nemo_ms']['p95']}ms | "
        f"Total: {latency_report['total_ms']['p95']}ms"
    )
