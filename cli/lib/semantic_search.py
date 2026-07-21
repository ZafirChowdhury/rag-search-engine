from sentence_transformers import SentenceTransformer

import os

import numpy as np
from numpy.typing import NDArray

from typing import Any

from .search_utils import Movie, MOVIE_EMBEDDINGS_PATH, DEFAULT_SEARCH_LIMIT

EmbeddingArray = NDArray[Any]

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        self.model = SentenceTransformer(model_name)
        self.max_seq_length = 256
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text: str):
        if len(text.strip()) == 0:
           raise ValueError("cannot generate embeding for a empty string")

        embedding = self.model.encode([text])
        return embedding[0]

    def build_embeddings(self, documents: list[Movie]):
        self.documents = documents
        self.document_map = {}
        movie_strings: list[str] =  []

        for doc in documents:
            self.document_map[doc["id"]] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")

        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)
        os.makedirs(os.path.dirname(MOVIE_EMBEDDINGS_PATH), exist_ok=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit=DEFAULT_SEARCH_LIMIT):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")

        query_embedding = self.generate_embedding(query)

        similarity_scores = []
        for i in range(len(self.embeddings)):
            cs = cosine_similarity(query_embedding, self.embeddings[i])
            similarity_scores.append((cs, self.document_map[i+1]))

        similarity_scores_sorted = sorted(similarity_scores, key=lambda x:x[0], reverse=True)

        # similarity_scores_sorted -> (score, Movie)
        results = []
        for score, doc in  similarity_scores_sorted[:limit]:
            results.append({
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"]
            })

        return results

# 1.0 -> same direction -> identical meaning
# 0.0 -> perpendicular -> unrelated
# -0.1 -> opposite directions -> opposite meaning
def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)
