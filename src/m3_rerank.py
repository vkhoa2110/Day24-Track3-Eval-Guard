from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time, re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._load_attempted = False

    def _load_model(self):
        if self._model is None and not self._load_attempted:
            self._load_attempted = True
            # Implemented cross-encoder model loading.
            # from sentence_transformers import CrossEncoder
            # self._model = CrossEncoder(self.model_name)
            #
            # ⚠️ LƯU Ý: Dùng sentence_transformers.CrossEncoder, KHÔNG dùng FlagEmbedding.
            # FlagReranker crash với transformers>=5.0 (XLMRobertaTokenizer lỗi).
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name, local_files_only=True)
            except Exception as e:
                print(f"  Warning: CrossEncoder unavailable, using lexical rerank fallback: {e}")
                self._model = None
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []

        model = self._load_model()
        if model is not None:
            try:
                pairs = [(query, doc.get("text", "")) for doc in documents]
                scores = model.predict(pairs)
                if isinstance(scores, (int, float)):
                    scores = [scores]
                scored = [(float(score), doc) for score, doc in zip(scores, documents)]
            except Exception as e:
                print(f"  Warning: CrossEncoder rerank failed, using lexical fallback: {e}")
                scored = self._lexical_scores(query, documents)
        else:
            scored = self._lexical_scores(query, documents)

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i + 1,
            )
            for i, (score, doc) in enumerate(scored[:top_k])
        ]

    def _lexical_scores(self, query: str, documents: list[dict]) -> list[tuple[float, dict]]:
        query_terms = set(re.findall(r"\w+", query.lower()))
        scored: list[tuple[float, dict]] = []
        for doc in documents:
            text = doc.get("text", "")
            doc_terms = set(re.findall(r"\w+", text.lower()))
            overlap = len(query_terms & doc_terms)
            lexical = overlap / max(len(query_terms), 1)
            score = lexical + 0.01 * float(doc.get("score", 0.0))
            scored.append((float(score), doc))
        return scored


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents:
            return []
        try:
            from flashrank import Ranker, RerankRequest

            if self._model is None:
                self._model = Ranker()
            passages = [
                {"id": i, "text": doc.get("text", ""), "meta": doc.get("metadata", {})}
                for i, doc in enumerate(documents)
            ]
            results = self._model.rerank(RerankRequest(query=query, passages=passages))[:top_k]
            return [
                RerankResult(
                    text=item.get("text", ""),
                    original_score=float(documents[item.get("id", 0)].get("score", 0.0)),
                    rerank_score=float(item.get("score", 0.0)),
                    metadata=item.get("meta", {}),
                    rank=i + 1,
                )
                for i, item in enumerate(results)
            ]
        except Exception:
            return CrossEncoderReranker().rerank(query, documents, top_k=top_k)


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
