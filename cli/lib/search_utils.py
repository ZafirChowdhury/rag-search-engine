import json
import os
from typing import TypedDict, Any


class Movie(TypedDict):
    id: int
    title: str
    description: str

class SearchResult(TypedDict):
    id: int
    title: str
    document: str
    score: float
    metadata: dict[str, Any]

# BM25
BM25_K1 = 1.5
BM25_B = 0.75

DEFAULT_SEARCH_LIMIT = 5

# __file__ -> current file
# os.path.dirname() -> drops the file name
# os.path.dirname() -> os.path.dirname() -> os.path.dirname() -> project root -> cli -> lib
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")

CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
MOVIE_EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, CACHE_DIR, "movie_embeddings.npy")


def load_movies() -> list[Movie]:
    with open(DATA_PATH, "r") as file:
        data = json.load(file)
    return data["movies"]

def format_search_result(
    doc_id: int, title: str, document: str, score: float, **metadata: Any
) -> SearchResult:

    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, 2),
        "metadata": metadata if metadata else {},
    }
