from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys, json, re
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    # Implemented chunk summarization.
    # if OPENAI_API_KEY:
    #     try:
    #         from openai import OpenAI
    #         client = OpenAI()
    #         resp = client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=[
    #                 {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
    #                 {"role": "user", "content": text},
    #             ],
    #             max_tokens=150,
    #         )
    #         return resp.choices[0].message.content.strip()
    #     except Exception as e:
    #         print(f"  ⚠️  OpenAI summarize failed: {e}")
    #
    # Extractive fallback (không cần API):
    # sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]
    # return ". ".join(sentences[:2]) + "." if sentences else text
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Summarize the text in 2 short Vietnamese sentences. Return only the summary.",
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"  Warning: OpenAI summarize failed, using fallback: {e}")

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    if not sentences:
        return text.strip()
    summary = " ".join(sentences[:2]).strip()
    return summary if summary.endswith((".", "!", "?")) else f"{summary}."


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    # Implemented HyQA generation.
    # if OPENAI_API_KEY:
    #     try:
    #         from openai import OpenAI
    #         client = OpenAI()
    #         resp = client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=[
    #                 {"role": "system", "content": f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng."},
    #                 {"role": "user", "content": text},
    #             ],
    #             max_tokens=200,
    #         )
    #         questions = resp.choices[0].message.content.strip().split("\n")
    #         return [q.strip().lstrip("0123456789.-) ") for q in questions if q.strip()][:n_questions]
    #     except Exception as e:
    #         print(f"  ⚠️  OpenAI HyQA failed: {e}")
    #
    # Extractive fallback:
    # import re
    # sentences = [s.strip() for s in re.split(r'[.!?\n]', text) if len(s.strip()) > 10]
    # return [f"{s.rstrip('.')}?" for s in sentences[:n_questions]]
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Create {n_questions} Vietnamese questions that can be answered "
                            "from the text. Return one question per line."
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
            )
            raw_questions = (resp.choices[0].message.content or "").strip().splitlines()
            questions = [
                q.strip().lstrip("0123456789.-) ").strip()
                for q in raw_questions
                if q.strip()
            ]
            return questions[:n_questions]
        except Exception as e:
            print(f"  Warning: OpenAI HyQA failed, using fallback: {e}")

    sentences = [s.strip(" .!?") for s in re.split(r"[.!?\n]+", text) if len(s.strip()) > 10]
    questions: list[str] = []
    if sentences:
        questions.append(f"Thong tin nao duoc neu ve {sentences[0][:80]}?")
    if re.search(r"\d+", text):
        questions.append("Quy dinh nay bao nhieu ngay hoac moc thoi gian nao?")
    if len(questions) < n_questions:
        questions.append("Doan van nay tra loi cho chinh sach nao?")
    return questions[:n_questions]


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    # Implemented contextual prepend.
    # if OPENAI_API_KEY:
    #     try:
    #         from openai import OpenAI
    #         client = OpenAI()
    #         resp = client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=[
    #                 {"role": "system", "content": "Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu."},
    #                 {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
    #             ],
    #             max_tokens=80,
    #         )
    #         context = resp.choices[0].message.content.strip()
    #         return f"{context}\n\n{text}"
    #     except Exception as e:
    #         print(f"  ⚠️  OpenAI contextual failed: {e}")
    #
    # Simple fallback:
    # prefix = f"Trích từ {document_title}. " if document_title else ""
    # return f"{prefix}{text}"
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Write one short Vietnamese context sentence explaining where this "
                            "chunk belongs in the document. Return only that sentence."
                        ),
                    },
                    {"role": "user", "content": f"Document: {document_title}\n\nChunk:\n{text}"},
                ],
                max_tokens=80,
            )
            context = (resp.choices[0].message.content or "").strip()
            return f"{context}\n\n{text}" if context else text
        except Exception as e:
            print(f"  Warning: OpenAI contextual failed, using fallback: {e}")

    prefix = f"Trich tu tai lieu {document_title}. " if document_title else "Ngu canh tai lieu. "
    return f"{prefix}{text}"


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    # Implemented auto metadata extraction.
    # if OPENAI_API_KEY:
    #     try:
    #         import json as _json
    #         from openai import OpenAI
    #         client = OpenAI()
    #         resp = client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=[
    #                 {"role": "system", "content": 'Trích xuất metadata từ đoạn văn. Trả về JSON: {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}'},
    #                 {"role": "user", "content": text},
    #             ],
    #             max_tokens=150,
    #         )
    #         return _json.loads(resp.choices[0].message.content)
    #     except Exception as e:
    #         print(f"  ⚠️  OpenAI metadata failed: {e}")
    #
    # return {"topic": "general", "entities": [], "category": "policy", "language": "vi"}
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            'Extract metadata as JSON with keys "topic", "entities", '
                            '"category", and "language". Return JSON only.'
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
            )
            content = (resp.choices[0].message.content or "").strip()
            content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.IGNORECASE | re.MULTILINE).strip()
            data = json.loads(content)
            if isinstance(data, dict):
                return data
        except Exception as e:
            print(f"  Warning: OpenAI metadata failed, using fallback: {e}")

    lower = text.lower()
    if any(k in lower for k in ["vpn", "mat khau", "password", "mfa", "wireguard"]):
        category = "it"
    elif any(k in lower for k in ["luong", "thuong", "phu cap", "chi phi", "expense"]):
        category = "finance"
    elif any(k in lower for k in ["nghi", "thu viec", "dao tao", "mentor", "nhan vien"]):
        category = "hr"
    else:
        category = "policy"

    entities = sorted(set(re.findall(r"\b[A-Z][A-Za-z0-9_-]{2,}\b|\b20\d{2}\b", text)))[:8]
    topic_source = summarize_chunk(text)
    topic = topic_source[:120].strip()
    return {"topic": topic, "entities": entities, "category": category, "language": "vi"}


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    # Implemented combined enrichment (1 call/chunk).
    # if OPENAI_API_KEY:
    #     try:
    #         import json as _json
    #         from openai import OpenAI
    #         client = OpenAI()
    #         resp = client.chat.completions.create(
    #             model="gpt-4o-mini",
    #             messages=[
    #                 {"role": "system", "content": """Phân tích đoạn văn và trả về JSON:
    # {
    #   "summary": "tóm tắt 2-3 câu",
    #   "questions": ["câu hỏi 1", "câu hỏi 2", "câu hỏi 3"],
    #   "context": "1 câu mô tả đoạn văn nằm ở đâu trong tài liệu",
    #   "metadata": {"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}
    # }"""},
    #                 {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
    #             ],
    #             max_tokens=400,
    #         )
    #         return _json.loads(resp.choices[0].message.content)
    #     except Exception as e:
    #         print(f"  ⚠️  Enrichment API failed: {e}")
    if OPENAI_API_KEY:
        try:
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Analyze the chunk and return JSON only with keys: "
                            "summary, questions, context, metadata. "
                            "metadata must include topic, entities, category, language."
                        ),
                    },
                    {"role": "user", "content": f"Document: {source}\n\nChunk:\n{text}"},
                ],
                max_tokens=400,
            )
            content = (resp.choices[0].message.content or "").strip()
            content = re.sub(r"^```(?:json)?|```$", "", content, flags=re.IGNORECASE | re.MULTILINE).strip()
            data = json.loads(content)
            if isinstance(data, dict):
                return {
                    "summary": str(data.get("summary", "")),
                    "questions": list(data.get("questions", []) or []),
                    "context": str(data.get("context", "")),
                    "metadata": dict(data.get("metadata", {}) or {}),
                }
        except Exception as e:
            print(f"  Warning: enrichment API failed, using fallback: {e}")

    context = f"Trich tu tai lieu {source}." if source else "Ngu canh tai lieu."
    return {
        "summary": summarize_chunk(text),
        "questions": generate_hypothesis_questions(text),
        "context": context,
        "metadata": extract_metadata(text),
    }


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enriched_text = f"{context_line}\n\n{text}" if context_line else text
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
