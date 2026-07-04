import os
import pickle
from collections import defaultdict

from .keyword_search import tokenize_text
from .search_utils import CACHE_DIR, load_movies


class InvertedIndex:
    def __init__(self) -> None:
        self.index: defaultdict[str, set[int]] = defaultdict(set)
        self.docmap = {}
        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)
        tokens = set(tokens)
        for token in tokens:
            self.index[token].add(doc_id)

    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            doc_id = movie["id"]
            self.docmap[doc_id] = movie

            doc_description = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)

        with open(self.index_path, "wb") as file:
            pickle.dump(self.index, file)

        with open(self.docmap_path, "wb") as file:
            pickle.dump(self.docmap, file)

    def get_documents(self, term: str) -> list[int]:
        return sorted(list(self.index[term]))


def build_inverted_index() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()
    movies = idx.get_documents("merida")
    print(f"First document for token 'merida' = {movies[0]}")
