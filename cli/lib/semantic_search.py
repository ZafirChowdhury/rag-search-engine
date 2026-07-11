from sentence_transformers import SentenceTransformer

import os

import numpy as np
from numpy.typing import NDArray

from typing import Any

from .search_utils import Movie, MOVIE_EMBEDDINGS_PATH

EmbeddingArray = NDArray[Any]

class SemanticSearch:
    def __init__(self) -> None:
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.max_seq_length = 256
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
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
