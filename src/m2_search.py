from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys, math, re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    # Implemented Vietnamese word segmentation.
    # 1. from underthesea import word_tokenize
    # 2. segmented = word_tokenize(text, format="text")
    # 3. return segmented.replace("_", " ")
    #
    # ⚠️ LƯU Ý: underthesea nối từ ghép bằng "_" (VD: "nghỉ_phép").
    # BM25 tokenize bằng split(" ") → "nghỉ_phép" thành 1 token,
    # nhưng query "nghỉ phép" thành 2 token → KHÔNG khớp.
    # Phải replace("_", " ") để BM25 hoạt động đúng.
    try:
        from underthesea import word_tokenize

        return word_tokenize(text, format="text").replace("_", " ")
    except Exception:
        return text


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        # Implemented BM25 indexing.
        # 1. self.documents = chunks
        # 2. For each chunk: segment_vietnamese(chunk["text"]) → split by space
        # 3. self.corpus_tokens = [tokenized list for each chunk]
        # 4. from rank_bm25 import BM25Okapi
        #    self.bm25 = BM25Okapi(self.corpus_tokens)
        self.documents = list(chunks)
        self.corpus_tokens = [
            segment_vietnamese(chunk.get("text", "")).lower().split()
            for chunk in self.documents
        ]
        try:
            from rank_bm25 import BM25Okapi

            self.bm25 = BM25Okapi(self.corpus_tokens)
        except Exception:
            self.bm25 = None

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        # Implemented BM25 search.
        # 1. if self.bm25 is None: return []
        # 2. tokenized_query = segment_vietnamese(query).split()
        # 3. scores = self.bm25.get_scores(tokenized_query)
        # 4. top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        # 5. Return [SearchResult(text=..., score=..., metadata=..., method="bm25")]
        #    Lọc scores[i] > 0 để bỏ docs không liên quan.
        if not self.documents:
            return []

        tokenized_query = segment_vietnamese(query).lower().split()
        if not tokenized_query:
            return []

        if self.bm25 is not None:
            scores = list(self.bm25.get_scores(tokenized_query))
        else:
            query_terms = set(tokenized_query)
            scores = [
                float(len(query_terms & set(tokens)) / max(len(query_terms), 1))
                for tokens in self.corpus_tokens
            ]

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results: list[SearchResult] = []
        for i in top_indices:
            score = float(scores[i])
            if score <= 0:
                continue
            doc = self.documents[i]
            results.append(
                SearchResult(
                    text=doc.get("text", ""),
                    score=score,
                    metadata=doc.get("metadata", {}),
                    method="bm25",
                )
            )
        return results


class DenseSearch:
    def __init__(self):
        self.client = None
        self._fallback_documents: list[dict] = []
        try:
            from qdrant_client import QdrantClient

            self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        except Exception as e:
            print(f"  Warning: Qdrant client unavailable, dense fallback enabled: {e}")
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        # Implemented dense indexing.
        # 1. from qdrant_client.models import Distance, VectorParams, PointStruct
        # 2. self.client.recreate_collection(collection, vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE))
        # 3. texts = [c["text"] for c in chunks]
        # 4. vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        # 5. points = [PointStruct(id=i, vector=v.tolist(), payload={**c.get("metadata", {}), "text": c["text"]}) ...]
        # 6. self.client.upsert(collection, points)
        self._fallback_documents = list(chunks)
        if not chunks or self.client is None:
            return
        try:
            from qdrant_client.models import Distance, PointStruct, VectorParams

            texts = [c.get("text", "") for c in chunks]
            vectors = self._get_encoder().encode(texts, show_progress_bar=True)
            vector_size = len(vectors[0]) if len(vectors) else EMBEDDING_DIM
            self.client.recreate_collection(
                collection,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            points = [
                PointStruct(
                    id=i,
                    vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
                    payload={**chunks[i].get("metadata", {}), "text": texts[i]},
                )
                for i, vector in enumerate(vectors)
            ]
            self.client.upsert(collection, points)
        except Exception as e:
            print(f"  Warning: dense indexing failed, using lexical fallback: {e}")

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        # Implemented dense search.
        # 1. query_vector = self._get_encoder().encode(query).tolist()
        # 2. response = self.client.query_points(collection, query=query_vector, limit=top_k)
        # 3. Return [SearchResult(text=pt.payload["text"], score=pt.score, metadata=pt.payload, method="dense")
        #            for pt in response.points]
        #
        # ⚠️ LƯU Ý: qdrant-client >= 2.0 dùng query_points(), KHÔNG phải search().
        if self.client is not None:
            try:
                query_vector = self._get_encoder().encode(query).tolist()
                response = self.client.query_points(collection, query=query_vector, limit=top_k)
                return [
                    SearchResult(
                        text=pt.payload.get("text", ""),
                        score=float(pt.score),
                        metadata={k: v for k, v in pt.payload.items() if k != "text"},
                        method="dense",
                    )
                    for pt in response.points
                ]
            except Exception as e:
                print(f"  Warning: dense search failed, using lexical fallback: {e}")

        return self._fallback_search(query, top_k)

    def _fallback_search(self, query: str, top_k: int) -> list[SearchResult]:
        query_terms = set(re.findall(r"\w+", segment_vietnamese(query).lower()))
        if not query_terms:
            return []
        scored: list[tuple[float, dict]] = []
        for doc in self._fallback_documents:
            text = doc.get("text", "")
            doc_terms = set(re.findall(r"\w+", segment_vietnamese(text).lower()))
            if not doc_terms:
                continue
            overlap = len(query_terms & doc_terms)
            score = overlap / math.sqrt(len(query_terms) * len(doc_terms))
            if score > 0:
                scored.append((float(score), doc))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            SearchResult(
                text=doc.get("text", ""),
                score=score,
                metadata=doc.get("metadata", {}),
                method="dense",
            )
            for score, doc in scored[:top_k]
        ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    # Implemented RRF.
    # 1. rrf_scores = {}  # text → {"score": float, "result": SearchResult}
    # 2. For each result_list in results_list:
    #      For rank, result in enumerate(result_list):
    #        if result.text not in rrf_scores: rrf_scores[result.text] = {"score": 0.0, "result": result}
    #        rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)
    # 3. Sort by score descending
    # 4. Return top_k SearchResult with method="hybrid"
    rrf_scores: dict[str, dict] = {}
    for results in results_list:
        for rank, result in enumerate(results):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {"score": 0.0, "result": result}
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)

    fused = sorted(rrf_scores.values(), key=lambda item: item["score"], reverse=True)
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=item["result"].metadata,
            method="hybrid",
        )
        for item in fused[:top_k]
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
