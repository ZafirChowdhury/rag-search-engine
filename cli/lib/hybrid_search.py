from numpy import number
import os

from .keyword_search import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch

class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query: str, limit: int):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query: str, alpha: float, limit: int = 5) -> list[dict]:
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")

    def rrf_search(self, query: str, k: int, limit: int = 10) -> list[dict]:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def min_max_normalization(numbers: list) -> list:
    if not numbers or len(numbers) == 0:
        return []

    mn = min(numbers)
    mx = max(numbers)
    normalized_score = []

    if mn == mx:
        return [1.0 for _ in range(len(numbers))]

    for score in numbers:
        n_score = (score - mn) / (mx - mn)
        normalized_score.append(n_score)

    return normalized_score
