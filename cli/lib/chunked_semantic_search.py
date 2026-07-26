import re
import os
import json
import numpy as np
from typing import TypedDict, Any
from numpy.typing import NDArray

from .semantic_search import SemanticSearch, cosine_similarity
from .search_utils import load_movies, format_search_result, SearchResult, CHUNK_EMBEDDINGS_PATH, CHUNK_METADATA_PATH, DEFAULT_SEMANTIC_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP, DOCUMENT_PREVIEW_LENGTH

class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int

EmbeddingArray = NDArray[Any]

def semantic_chunk(text: str, max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []

    i = 0
    n_sentences = len(sentences)
    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
        i += max_chunk_size - overlap

    return chunks

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
         super().__init__(model_name)
         self.chunk_embeddings = []
         self.chunk_metadata = []

    def build_chunk_embeddings(self, documents) -> np.ndarray:
        self.documents = documents

        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks: list[str] = []
        chunk_metadata: list[ChunkMetadata] = []

        for idx, doc in enumerate(documents):
            text = doc.get("description", "")
            if not text.strip():
                continue

            chunks = semantic_chunk(text)

            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {"movie_idx": idx, "chunk_idx": i, "total_chunks": len(chunks)}
                )

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)

        with open(CHUNK_METADATA_PATH, "w") as f:
                json.dump(
                    {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
                )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(CHUNK_METADATA_PATH):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)

            with open(CHUNK_METADATA_PATH, "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]

            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        movies = load_movies()
        self.load_or_create_chunk_embeddings(movies)

        chunk_score_list = []

        qurey_embeding = self.generate_embedding(query)
        for i, chunk in enumerate(self.chunk_embeddings):
            cs = cosine_similarity(chunk, qurey_embeding)
            chunk_score_list.append({
                "chunk_idx": self.chunk_metadata[i]["chunk_idx"],
                "movie_idx": self.chunk_metadata[i]["movie_idx"],
                "score": cs
            })

        movie_scores = {}
        for chunk_score in chunk_score_list:
            movie_idx = chunk_score["movie_idx"]

            if (
                movie_idx not in movie_scores
                or chunk_score["score"] > movie_scores[movie_idx]
            ):
                movie_scores[movie_idx] = chunk_score["score"]

        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

        if self.documents is None:
            raise ValueError("No documents loaded. Call load_or_create_chunk_embeddings first.")

        results: list[SearchResult] = []
        for movie_idx, score in sorted_movies[:limit]:
            if movie_idx is None:
                continue
            doc = self.documents[movie_idx]
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"][:DOCUMENT_PREVIEW_LENGTH],
                    score=score,
                )
            )

        return results
