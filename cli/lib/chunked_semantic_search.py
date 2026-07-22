import re
import os
import json
import numpy as np
from typing import TypedDict, Any
from numpy.typing import NDArray

from semantic_search import SemanticSearch
from search_utils import CHUNK_EMBEDDINGS_PATH, CHUNK_METADATA_PATH

class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int

EmbeddingArray = NDArray[Any]

def semantic_chunk(text: str, max_chunk_size=4, overlap=0) -> list:
    sentences = re.split(r"(?<=[.!?])\s+", text)

    chunks = []
    step = max_chunk_size - overlap
    if step <= 0:
        raise ValueError("Chunk size must be bigger then overlap")

    pointer = 0
    while pointer < len(sentences):
        chunk = " ".join(sentences[pointer : pointer + max_chunk_size])
        chunks.append(chunk)

        pointer += step

    return chunks

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
         super().__init__(model_name)
         self.chunk_embeddings = None
         self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
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
                ) # why do i need to store total number of chunks?? in every chunk metadata??

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)

        with open(CHUNK_METADATA_PATH, "w") as f:
                json.dump(
                    {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
                ) # why use json for chunk metadata??

        return self.chunk_embeddings
