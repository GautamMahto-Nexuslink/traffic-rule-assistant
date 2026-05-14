import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


class HybridRetriever:
    """
    Combines BM25 keyword search and FAISS semantic search.

    Both score arrays are min-max normalised to [0, 1] then blended:
        score = alpha * bm25_norm + (1 - alpha) * faiss_norm

    alpha=1.0 → pure BM25, alpha=0.0 → pure FAISS, alpha=0.5 → balanced.
    """

    def __init__(
        self,
        chunks: list[str],
        index: faiss.IndexFlatL2,
        model: SentenceTransformer,
        alpha: float = 0.5,
    ) -> None:
        self.chunks = chunks
        self.index = index
        self.model = model
        self.alpha = alpha

        tokenized = [chunk.lower().split() for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized)

    # ── internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _normalise(arr: np.ndarray) -> np.ndarray:
        lo, hi = arr.min(), arr.max()
        return (arr - lo) / (hi - lo + 1e-9)

    def _bm25_scores(self, query: str) -> np.ndarray:
        return self.bm25.get_scores(query.lower().split())

    def _faiss_scores(self, query: str) -> np.ndarray:
        """Convert L2 distances to similarity scores for all chunks."""
        n = len(self.chunks)
        query_vec = self.model.encode([query]).astype(np.float32)
        distances, indices = self.index.search(query_vec, n)

        scores = np.zeros(n, dtype=np.float32)
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < n:
                scores[idx] = 1.0 / (1.0 + dist)  # lower distance → higher score
        return scores

    # ── public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str, top_k: int = 5) -> list[str]:
        bm25_norm = self._normalise(self._bm25_scores(query))
        faiss_norm = self._normalise(self._faiss_scores(query))

        combined = self.alpha * bm25_norm + (1.0 - self.alpha) * faiss_norm
        top_indices = np.argsort(combined)[::-1][:top_k]
        return [self.chunks[i] for i in top_indices]
